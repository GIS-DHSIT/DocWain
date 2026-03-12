from __future__ import annotations

import concurrent.futures
import json
from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass
from typing import Iterable, Optional

from .types import GenericSchema, HRSchema, InvoiceSchema, LegalSchema, LLMBudget, LLMResponseSchema, MultiEntitySchema

logger = get_logger(__name__)

@dataclass
class JudgeResult:
    status: str
    reason: str = ""
    confidence: float = 0.0  # 0.0-1.0 graduated confidence score

FALLBACK_STATUS = JudgeResult(status="fail", reason="fallback")

_ENTITY_NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2}\b")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")

_STOPWORDS = frozenset({
    "the", "is", "in", "at", "of", "and", "or", "to", "for", "an",
    "it", "on", "by", "be", "as", "do", "if", "so", "no", "not",
    "are", "was", "were", "has", "had", "have", "been", "from",
    "this", "that", "with", "what", "which", "who", "how", "can",
    "will", "does", "did", "its", "my", "me", "we", "our", "your",
    "their", "them", "there", "here", "about", "would", "should",
    "could", "may", "any", "all", "each", "more", "some", "than",
})

# ── Synonym pairs for answer-relevance matching ──────────────────────────
_QUERY_ANSWER_SYNONYMS: dict[str, frozenset[str]] = {
    "salary": frozenset({"compensation", "pay", "wage", "wages", "remuneration", "earnings", "income"}),
    "compensation": frozenset({"salary", "pay", "wage", "wages", "remuneration", "earnings", "income"}),
    "experience": frozenset({"background", "work history", "tenure", "career"}),
    "skills": frozenset({"competencies", "abilities", "expertise", "proficiency", "qualifications"}),
    "education": frozenset({"degree", "qualification", "academic", "university", "school", "diploma"}),
    "cost": frozenset({"price", "amount", "charge", "fee", "expense", "total"}),
    "price": frozenset({"cost", "amount", "charge", "fee", "expense", "rate"}),
    "employee": frozenset({"worker", "staff", "personnel", "associate", "team member"}),
    "company": frozenset({"organization", "firm", "corporation", "enterprise", "business"}),
    "address": frozenset({"location", "residence", "office", "headquarters"}),
    "phone": frozenset({"telephone", "contact number", "mobile", "cell"}),
    "email": frozenset({"mail", "email address", "contact"}),
    "deadline": frozenset({"due date", "expiration", "expiry", "cutoff"}),
    "benefit": frozenset({"perk", "advantage", "allowance", "entitlement"}),
    "risk": frozenset({"hazard", "threat", "liability", "danger", "exposure"}),
    "treatment": frozenset({"therapy", "medication", "procedure", "intervention"}),
    "diagnosis": frozenset({"condition", "finding", "assessment", "evaluation"}),
    "patient": frozenset({"client", "individual", "subject"}),
    "clause": frozenset({"provision", "section", "article", "term", "stipulation"}),
    "contract": frozenset({"agreement", "arrangement", "deal"}),
    "summary": frozenset({"overview", "synopsis", "abstract", "brief", "recap"}),
    "candidate": frozenset({"applicant", "prospect", "nominee", "individual"}),
    "revenue": frozenset({"income", "sales", "earnings", "turnover"}),
    "department": frozenset({"division", "unit", "team", "group", "section"}),
}

# ── Unit multiplier patterns for numeric normalization ────────────────────
_UNIT_MULTIPLIERS: dict[str, float] = {
    "k": 1_000, "K": 1_000,
    "m": 1_000_000, "M": 1_000_000,
    "b": 1_000_000_000, "B": 1_000_000_000,
    "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000,
}

# Regex to capture numbers with optional currency prefix and unit suffix
_UNIT_NUMBER_RE = re.compile(
    r"(?:\$|€|£|₹)?\s*(\d+(?:[.,]\d+)?)\s*"
    r"(k|K|m|M|b|B|thousand|million|billion)?\b"
)

# Regex to capture range patterns like "40K-60K", "40,000 - 60,000"
_RANGE_RE = re.compile(
    r"(?:\$|€|£|₹)?\s*(\d+(?:[.,]\d+)?)\s*(k|K|m|M|b|B|thousand|million|billion)?\s*"
    r"[-–—to]+\s*"
    r"(?:\$|€|£|₹)?\s*(\d+(?:[.,]\d+)?)\s*(k|K|m|M|b|B|thousand|million|billion)?\b"
)

# Percentage word equivalents
_PERCENTAGE_WORDS: dict[str, float] = {
    "half": 50.0, "quarter": 25.0, "third": 33.33,
    "three quarters": 75.0, "three-quarters": 75.0,
    "two thirds": 66.67, "two-thirds": 66.67,
}

# ── Title prefixes for entity alias matching ──────────────────────────────
_TITLE_PREFIXES = frozenset({
    "mr", "mrs", "ms", "miss", "dr", "prof", "professor",
    "sir", "madam", "lord", "lady",
})

# ── Implicit entity reference words ──────────────────────────────────────
_IMPLICIT_REFERENCES = frozenset({
    "the candidate", "the applicant", "the patient", "the client",
    "the employee", "the worker", "the individual", "the person",
    "the subject", "the vendor", "the supplier", "the contractor",
    "this candidate", "this applicant", "this patient", "this person",
    "he", "she", "they", "his", "her", "their",
})

def _tokenize(text: str) -> list[str]:
    """Lowercase tokenize, dropping stopwords and very short tokens."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]

# ── Numeric normalization helpers ─────────────────────────────────────────

def _normalize_number(num_str: str, suffix: str | None = None) -> float:
    """Normalize a number string with optional unit suffix to a float."""
    cleaned = num_str.replace(",", "")
    try:
        value = float(cleaned)
    except ValueError:
        return 0.0
    if suffix and suffix in _UNIT_MULTIPLIERS:
        value *= _UNIT_MULTIPLIERS[suffix]
    return value

def _extract_all_normalized_numbers(text: str) -> set[float]:
    """Extract all numbers from text, normalizing units (e.g. $50K → 50000)."""
    results: set[float] = set()
    for match in _UNIT_NUMBER_RE.finditer(text):
        num_str, suffix = match.group(1), match.group(2)
        results.add(_normalize_number(num_str, suffix))
    return results

def _extract_ranges(text: str) -> list[tuple[float, float]]:
    """Extract numeric ranges from text (e.g. '40K-60K' → (40000, 60000))."""
    ranges: list[tuple[float, float]] = []
    for match in _RANGE_RE.finditer(text):
        lo = _normalize_number(match.group(1), match.group(2))
        hi = _normalize_number(match.group(3), match.group(4))
        if lo <= hi:
            ranges.append((lo, hi))
        else:
            ranges.append((hi, lo))
    return ranges

def _number_in_any_range(num: float, ranges: list[tuple[float, float]]) -> bool:
    """Check if a number falls within any of the given ranges."""
    return any(lo <= num <= hi for lo, hi in ranges)

def _normalize_percentage(text: str) -> set[float]:
    """Extract percentage values from text, normalizing words and decimals.

    '50%' → {50.0}, '0.50' → {50.0}, 'half' → {50.0}
    """
    results: set[float] = set()
    # Explicit percentage: "50%", "33.5%"
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", text):
        results.add(float(m.group(1)))
    # Decimal fractions that look like percentages: "0.50" → 50.0, "0.25" → 25.0
    for m in re.finditer(r"\b0\.(\d{1,2})\b", text):
        digits = m.group(1)
        results.add(float(digits) * (10 if len(digits) == 1 else 1))
    # Word equivalents
    lowered = text.lower()
    for word, pct in _PERCENTAGE_WORDS.items():
        if word in lowered:
            results.add(pct)
    return results

# ── Entity alias helpers ──────────────────────────────────────────────────

def _build_entity_aliases(entities: set[str]) -> dict[str, set[str]]:
    """Build alias groups from full entity names.

    "John Smith" produces aliases: {"john smith", "john", "mr. smith",
    "mr smith", "j. smith", "j smith", "smith"}
    """
    alias_map: dict[str, set[str]] = {}
    for entity in entities:
        parts = entity.split()
        canon = entity.lower()
        aliases: set[str] = {canon}
        if len(parts) >= 2:
            first = parts[0].lower()
            last = parts[-1].lower()
            # Skip if first part is a title
            if first in _TITLE_PREFIXES and len(parts) >= 3:
                first = parts[1].lower()
                last = parts[-1].lower()
            aliases.add(first)  # "John"
            aliases.add(last)   # "Smith"
            aliases.add(f"mr. {last}")
            aliases.add(f"mr {last}")
            aliases.add(f"mrs. {last}")
            aliases.add(f"mrs {last}")
            aliases.add(f"ms. {last}")
            aliases.add(f"ms {last}")
            aliases.add(f"dr. {last}")
            aliases.add(f"dr {last}")
            # Initial + last: "J. Smith", "J Smith"
            if first and first[0].isalpha():
                aliases.add(f"{first[0]}. {last}")
                aliases.add(f"{first[0]} {last}")
        alias_map[canon] = aliases
    return alias_map

def _entity_mentioned_in_text(aliases: set[str], text_lower: str) -> bool:
    """Check if any alias of an entity appears in the given text."""
    return any(alias in text_lower for alias in aliases)

def _check_entity_consistency(
    query: str, answer: str, intent: str,
    evidence_texts: Optional[list[str]] = None,
) -> Optional[str]:
    """Check that answer entities are consistent with query and evidence.

    Returns a warning string if mismatch detected, None if OK.
    For multi-entity intents (comparison, ranking), validates entities
    against evidence rather than just the query.
    """
    query_entities = set(_ENTITY_NAME_RE.findall(query))
    answer_entities = set(_ENTITY_NAME_RE.findall(answer))
    if not answer_entities:
        return None  # No entities in answer to check

    # For multi-entity intents, validate against evidence instead of query
    if intent in ("comparison", "ranking", "cross_document"):
        if not evidence_texts:
            return None
        evidence_combined = " ".join(evidence_texts)
        evidence_entities = set(_ENTITY_NAME_RE.findall(evidence_combined))

        # Build aliases for evidence entities
        evidence_aliases = _build_entity_aliases(evidence_entities)
        all_evidence_aliases: set[str] = set()
        for aliases in evidence_aliases.values():
            all_evidence_aliases.update(aliases)

        answer_names = {e.lower() for e in answer_entities}
        foreign = set()
        for name in answer_names:
            if name not in all_evidence_aliases and len(name) > 3:
                # Check partial match (first name or last name)
                parts = name.split()
                if not any(p in all_evidence_aliases for p in parts):
                    foreign.add(name)

        if len(foreign) > 2:
            return f"multi_entity_mismatch: answer mentions {foreign} not found in evidence"

        # Cross-validate: if query mentions N entities, check answer covers them
        if query_entities and len(query_entities) >= 2:
            query_aliases = _build_entity_aliases(query_entities)
            answer_lower = answer.lower()
            missing = []
            for canon, aliases in query_aliases.items():
                if not _entity_mentioned_in_text(aliases, answer_lower):
                    missing.append(canon)
            if len(missing) > 0 and len(missing) >= len(query_entities) // 2:
                return f"multi_entity_incomplete: answer missing {missing} from query"

        return None

    if not query_entities:
        return None  # No specific entity in query

    # Build alias groups for query entities
    query_aliases = _build_entity_aliases(query_entities)
    all_query_aliases: set[str] = set()
    for aliases in query_aliases.values():
        all_query_aliases.update(aliases)

    answer_lower = answer.lower()

    # Check if answer mentions entities NOT in the query (using aliases)
    answer_names = {e.lower() for e in answer_entities}
    foreign = set()
    for name in answer_names:
        name_parts = name.split()
        # Check if any part of the answer entity matches any query alias
        matched = (
            name in all_query_aliases
            or any(p in all_query_aliases for p in name_parts)
        )
        if not matched:
            foreign.add(name)

    # Check implicit references: if answer uses "the candidate" etc.,
    # verify the query entity appears somewhere in evidence
    if foreign and evidence_texts:
        evidence_lower = " ".join(evidence_texts).lower()
        # If implicit references are used AND query entity is in evidence, allow it
        has_implicit = any(ref in answer_lower for ref in _IMPLICIT_REFERENCES)
        if has_implicit:
            # Verify the query entity is actually in the evidence
            for _canon, aliases in query_aliases.items():
                if _entity_mentioned_in_text(aliases, evidence_lower):
                    # Implicit reference is grounded — remove foreign entities
                    # that are just implicit references
                    foreign = {f for f in foreign if f not in _IMPLICIT_REFERENCES}
                    break

    if foreign and not any(
        _entity_mentioned_in_text(aliases, answer_lower)
        for aliases in query_aliases.values()
    ):
        return f"entity_mismatch: query about {query_entities} but answer mentions {foreign}"

    return None

def _check_numeric_fidelity(
    answer: str, evidence_texts: list[str],
) -> list[str]:
    """Check that numbers in the answer appear in the evidence.

    Supports unit normalization ($50K ↔ 50,000), range checking
    (answer "50K" valid if evidence says "40K-60K"), and percentage
    normalization (50% ↔ 0.50 ↔ "half").

    Returns list of hallucinated numbers.
    """
    answer_numbers = set(_NUMBER_RE.findall(answer))
    if not answer_numbers:
        return []

    evidence_text = " ".join(evidence_texts)
    evidence_numbers = set(_NUMBER_RE.findall(evidence_text))

    # Allow common numbers that don't need grounding (ordinals, small counts,
    # percentages, and common formatting values)
    trivial = {
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
        "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
        "30", "33", "40", "50", "60", "75", "90", "100",
        "0.5", "1.0", "2.0", "3.0", "4.0", "5.0",
        "24", "48", "72",  # common hour durations
        "7",   # days in week (already covered by single digits)
        "52",  # weeks in year
        "365", # days in year
        "000", # trailing fragment from comma-separated numbers (e.g. 50,000)
        "10%", "20%", "25%", "33%", "50%", "75%", "100%",
    }

    # Pre-compute normalized numbers from full text (captures "$50K", "50,000" etc.)
    answer_normalized = _extract_all_normalized_numbers(answer)
    evidence_normalized = _extract_all_normalized_numbers(evidence_text)
    evidence_ranges = _extract_ranges(evidence_text)
    evidence_percentages = _normalize_percentage(evidence_text)
    answer_percentages = _normalize_percentage(answer)

    # First pass: check if all normalized answer numbers are grounded in evidence
    # This catches unit conversions like "$50K" ↔ "50,000" at the full-text level
    ungrounded_normalized = set()
    for ans_val in answer_normalized:
        if ans_val == 0.0:
            continue  # Skip zero
        # Direct normalized match
        if ans_val in evidence_normalized:
            continue
        # Range match
        if evidence_ranges and _number_in_any_range(ans_val, evidence_ranges):
            continue
        # Close match with tolerance (handles rounding: 49,999 vs 50,000)
        if any(abs(ans_val - ev) / max(ev, 1) < 0.01 for ev in evidence_normalized if ev > 0):
            continue
        ungrounded_normalized.add(ans_val)

    # Second pass: per raw number, check trivial/evidence/normalized
    hallucinated = []
    for num in answer_numbers:
        if num in trivial:
            continue
        if num in evidence_numbers:
            continue

        # Check if this raw number's normalized form is grounded
        raw_normalized = _extract_all_normalized_numbers(num)
        if raw_normalized:
            raw_val = next(iter(raw_normalized))
            if raw_val not in ungrounded_normalized:
                continue  # Grounded via normalized full-text check

        # Percentage normalization: "50%" matches "0.50" matches "half"
        if num.endswith("%"):
            pct_str = num.rstrip("%")
            try:
                pct_val = float(pct_str)
                if pct_val in evidence_percentages:
                    continue
            except ValueError:
                pass
        # Check if this number appears as a percentage in evidence
        if answer_percentages & evidence_percentages:
            plain = num.rstrip("%")
            try:
                if float(plain) in evidence_percentages:
                    continue
            except ValueError:
                pass

        hallucinated.append(num)

    return hallucinated

def _check_table_integrity(answer: str) -> str:
    """Check markdown table integrity — column consistency and completeness.

    Returns a warning string if issues found, empty string if table is valid.
    Tables with inconsistent column counts produce garbled rendering.
    """
    if "|" not in answer:
        return ""

    lines = answer.split("\n")
    table_lines = [l.strip() for l in lines if l.strip().startswith("|") and l.strip().endswith("|")]
    if len(table_lines) < 2:
        return ""  # Not a real table

    # Count pipes in each row (columns = pipes - 1)
    col_counts = []
    for line in table_lines:
        # Skip separator rows
        if re.match(r"^\|[\s\-:]+\|$", line.replace("|", "| |").replace("  ", " ")):
            continue
        count = line.count("|") - 1  # First and last pipe don't count as separators
        if count > 0:
            col_counts.append(count)

    if not col_counts:
        return ""

    # Check consistency — all data rows should have same column count
    expected = col_counts[0]
    mismatched = [i for i, c in enumerate(col_counts) if c != expected]
    if len(mismatched) > len(col_counts) * 0.3:
        return f"table_column_mismatch: expected {expected} cols, {len(mismatched)}/{len(col_counts)} rows differ"

    return ""

def _check_answer_relevance(query: str, answer: str, intent: str = "") -> float:
    """Check if answer addresses the question topic. Returns relevance score 0.0-1.0.

    Uses semantic synonym matching and intent-aware thresholds.
    """
    if not query or not answer:
        return 0.5

    query_tokens = set(_tokenize(query))
    answer_tokens = set(_tokenize(answer))

    if not query_tokens:
        return 1.0

    # Direct token overlap
    overlap_count = len(query_tokens & answer_tokens)

    # Stem-like suffix matching: "qualification" ↔ "qualifications", "diagnose" ↔ "diagnosis"
    # Catches plural/tense variants without a full stemmer dependency
    stem_hits = 0
    _unmatched_query = query_tokens - answer_tokens
    for qt in _unmatched_query:
        # Check if any answer token shares a common prefix (≥5 chars)
        _prefix_len = max(5, len(qt) - 3)
        if any(at[:_prefix_len] == qt[:_prefix_len] for at in answer_tokens if len(at) >= _prefix_len):
            stem_hits += 1

    # Semantic synonym expansion: count query terms whose synonyms appear in answer
    synonym_hits = 0
    for qt in _unmatched_query:
        synonyms = _QUERY_ANSWER_SYNONYMS.get(qt, frozenset())
        if synonyms & answer_tokens:
            synonym_hits += 1

    total_matches = overlap_count + stem_hits + synonym_hits
    overlap = total_matches / len(query_tokens)

    # Boost if answer mentions key query nouns (longer tokens more likely to be meaningful)
    key_query_terms = {t for t in query_tokens if len(t) >= 4}
    if key_query_terms:
        key_direct = len(key_query_terms & answer_tokens)
        key_synonym = 0
        for kt in key_query_terms:
            if kt in answer_tokens:
                continue
            synonyms = _QUERY_ANSWER_SYNONYMS.get(kt, frozenset())
            if synonyms & answer_tokens:
                key_synonym += 1
        key_overlap = (key_direct + key_synonym) / len(key_query_terms)
        overlap = max(overlap, key_overlap)

    # Context mismatch penalty: if a key query term appears in the answer but
    # only as a modifier (e.g., "revenue uplift" vs "annual revenue"), penalize.
    # Detect by checking if the answer's first 200 chars contain the query's
    # subject phrase (2+ adjacent query terms) rather than scattered mentions.
    if key_query_terms and len(key_query_terms) >= 2 and overlap > 0.0:
        _ql = query.lower()
        _al_lead = answer[:300].lower()
        # Extract 2-word query phrases (adjacent pairs)
        _q_words = [w for w in _ql.split() if len(w) >= 4]
        _phrase_found = False
        for i in range(len(_q_words) - 1):
            _bigram = f"{_q_words[i]} {_q_words[i+1]}"
            if _bigram in _al_lead:
                _phrase_found = True
                break
        # If no query phrase found in answer lead, reduce overlap (scattered mention)
        if not _phrase_found and overlap < 0.5:
            overlap *= 0.7

    return round(min(1.0, overlap), 4)

def _get_relevance_threshold(intent: str) -> float:
    """Return the minimum relevance threshold based on intent type."""
    # Factual/extraction intents need higher overlap — the answer should
    # directly address the specific question.
    _HIGH_THRESHOLD_INTENTS = frozenset({
        "factual", "extraction", "extract", "qa", "contact", "list",
    })
    # Summary/analytical intents may legitimately use different vocabulary.
    _LOW_THRESHOLD_INTENTS = frozenset({
        "summary", "summarize", "overview", "analysis", "reasoning",
        "comparison", "ranking", "cross_document", "timeline", "generate",
    })
    if intent in _HIGH_THRESHOLD_INTENTS:
        return 0.30
    if intent in _LOW_THRESHOLD_INTENTS:
        return 0.15
    return 0.25  # default

# ── Factual consistency checking ──────────────────────────────────────────

def _check_source_citations(answer: str, evidence_texts: list[str]) -> Optional[str]:
    """Check if source citations in the answer match actual evidence sources.

    Detects fabricated source references like "(source: fake_file.pdf)"
    that don't correspond to any actual document in the evidence.
    Returns a warning string if fabricated sources found, None if OK.
    """
    if not answer or not evidence_texts:
        return None

    # Extract source citations from answer: (Source: X), Source: X
    _source_re = re.compile(
        r"\(\s*[Ss]ource\s*:\s*([^)]+)\s*\)",
    )
    cited_sources = []
    for m in _source_re.finditer(answer):
        source = m.group(1).strip().rstrip(".")
        if source and len(source) > 3:
            cited_sources.append(source.lower())

    if not cited_sources:
        return None

    # Build evidence source set from all evidence text
    evidence_combined = " ".join(evidence_texts).lower()

    fabricated = []
    for source in cited_sources:
        # Check if source name appears anywhere in evidence
        if source not in evidence_combined:
            # Try partial match (filename without extension)
            base = source.rsplit(".", 1)[0] if "." in source else source
            if base not in evidence_combined:
                fabricated.append(source)

    if fabricated and len(fabricated) / max(len(cited_sources), 1) > 0.20:
        return (
            f"fabricated_sources: {len(fabricated)}/{len(cited_sources)} "
            f"cited sources not found in evidence"
        )

    return None

def _check_factual_consistency(answer: str, evidence_texts: list[str]) -> Optional[str]:
    """Check if the answer makes claims not supported by evidence.

    Detects when specific entity+number pairs in the answer don't appear
    anywhere in the evidence. This catches hallucinated statistics.
    Returns a warning string if inconsistency found, None if OK.
    """
    if not answer or not evidence_texts or len(answer) < 30:
        return None

    evidence_combined = " ".join(evidence_texts).lower()

    # Extract entity+number claims from the answer
    # Pattern: "Name has/earned/scored/totals N" or "Name: $N"
    # Also handles titles like "Dr. Alice Johnson" and percentages
    _claim_re = re.compile(
        r"(?:\*\*)?([A-Z][a-zA-Z.\s]{1,35}?)(?:\*\*)?\s*"
        r"(?:has|had|earned|received|scored|totals?|:|is|was|were|gets?|makes?)\s*"
        r"(?:\*\*)?[\$£€₹]?(\d[\d,]*\.?\d*%?)(?:\*\*)?",
        re.IGNORECASE,
    )

    unsupported_claims = 0
    total_claims = 0

    for match in _claim_re.finditer(answer):
        entity = match.group(1).strip().lower()
        number = match.group(2).replace(",", "")
        if len(entity) < 3 or len(number) < 1:
            continue

        total_claims += 1

        # Check if this number appears near this entity in evidence
        entity_found = entity in evidence_combined
        number_found = number in evidence_combined

        if not number_found:
            unsupported_claims += 1

    # Only flag if >40% of specific claims are unsupported AND at least 2
    if total_claims >= 3 and unsupported_claims >= 2:
        ratio = unsupported_claims / total_claims
        if ratio > 0.40:
            return (
                f"factual_inconsistency: {unsupported_claims}/{total_claims} "
                f"entity-number claims not found in evidence"
            )

    return None

# ── Completeness checking ─────────────────────────────────────────────────

_COMPLETENESS_TRIGGERS = re.compile(
    r"\b(all|every|each|list\s+of|complete\s+list|full\s+list|entire)\b", re.IGNORECASE
)
_COUNT_TRIGGER = re.compile(
    r"\b(?:top|first|last|bottom)\s+(\d+)\b", re.IGNORECASE
)

def _check_answer_completeness(
    query: str, answer: str,
    evidence_texts: list[str],
    intent: str,
) -> Optional[str]:
    """Check if the answer covers the expected number of items.

    Returns a warning string if answer is incomplete, None if OK.
    """
    if not query or not answer:
        return None

    query_lower = query.lower()

    # Check for explicit count requests: "top 3", "first 5"
    count_match = _COUNT_TRIGGER.search(query_lower)
    if count_match:
        expected = int(count_match.group(1))
        if expected > 20:
            return None  # Unreasonable, skip

        # Count items in the answer (numbered list items, bullet items, table rows)
        answer_items = _count_answer_items(answer)
        if answer_items > 0 and answer_items < expected:
            return (
                f"incomplete: query asks for {expected} items "
                f"but answer has {answer_items}"
            )
        return None

    # Check for "all/every/each" completeness triggers
    if not _COMPLETENESS_TRIGGERS.search(query_lower):
        return None

    # Count entities in evidence vs answer
    evidence_combined = " ".join(evidence_texts)
    evidence_entity_set = set(_ENTITY_NAME_RE.findall(evidence_combined))
    answer_entity_set = set(_ENTITY_NAME_RE.findall(answer))

    # Only check if evidence has multiple entities
    if len(evidence_entity_set) < 2:
        return None

    # For each evidence entity, check if it (or an alias) appears in the answer
    answer_lower = answer.lower()
    covered = 0
    for entity in evidence_entity_set:
        aliases = _build_entity_aliases({entity})
        for _canon, alias_set in aliases.items():
            if _entity_mentioned_in_text(alias_set, answer_lower):
                covered += 1
                break

    if len(evidence_entity_set) >= 3:
        coverage_ratio = covered / len(evidence_entity_set)
        if coverage_ratio < 0.80:
            return (
                f"incomplete: answer covers {covered}/{len(evidence_entity_set)} "
                f"entities ({coverage_ratio:.0%}) from evidence"
            )

    return None

def _count_answer_items(answer: str) -> int:
    """Count discrete items in the answer (numbered, bulleted, or table rows)."""
    lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]

    # Numbered items: "1.", "2)", "1:"
    numbered = sum(1 for ln in lines if re.match(r"^\d+[.):\-]\s", ln))
    if numbered >= 2:
        return numbered

    # Bullet items
    bulleted = sum(1 for ln in lines if re.match(r"^[-*•]\s", ln))
    if bulleted >= 2:
        return bulleted

    # Table rows (lines with | separators, excluding header separator)
    table_rows = sum(
        1 for ln in lines
        if "|" in ln and not re.match(r"^[\s|:-]+$", ln)
    )
    if table_rows >= 2:
        # Subtract header row
        return max(table_rows - 1, 1)

    return 0

def judge_answer(
    *,
    answer: str,
    schema: InvoiceSchema | HRSchema | LegalSchema | GenericSchema | MultiEntitySchema,
    intent: str,
    llm_client: Optional[object],
    budget: LLMBudget,
    sources_present: bool = True,
    correlation_id: Optional[str] = None,
    query: str = "",
) -> JudgeResult:
    if not answer:
        return JudgeResult(status="fail", reason="empty_answer")
    if not sources_present:
        return JudgeResult(status="fail", reason="no_sources")

    heuristic = _heuristic_judge(answer, schema, intent, query)

    # Run critical fidelity checks on both "pass" and "uncertain" paths
    # Numeric fidelity and source citations catch hallucinations regardless of heuristic status
    if heuristic.status in ("pass", "uncertain") and query:
        evidence_texts = list(_iter_spans(schema))
        # Skip self-referential evidence (LLMResponseSchema yields its own text)
        _is_self_evidence = isinstance(schema, LLMResponseSchema)

        if not _is_self_evidence:
            # Numeric fidelity — catch hallucinated numbers
            if evidence_texts:
                hallucinated = _check_numeric_fidelity(answer, evidence_texts)
                if len(hallucinated) >= 3:
                    return JudgeResult(status="fail", reason=f"numeric_fidelity: {hallucinated[:3]}")
            # Source citation validation — detect fabricated references
            if evidence_texts:
                citation_issue = _check_source_citations(answer, evidence_texts)
                if citation_issue:
                    logger.warning("Judge source citations: %s", citation_issue)
                    return JudgeResult(status="uncertain", reason=citation_issue)

        # Full fidelity suite only on "pass" path
        if heuristic.status == "pass":
            if not _is_self_evidence:
                entity_warning = _check_entity_consistency(query, answer, intent, evidence_texts)
                if entity_warning:
                    logger.warning("Judge entity check: %s", entity_warning)
                    return JudgeResult(status="fail", reason=entity_warning)
                if evidence_texts:
                    # Temporal consistency check
                    temporal_warning = _check_temporal_consistency(answer, evidence_texts)
                    if temporal_warning:
                        logger.warning("Judge temporal check: %s", temporal_warning)
                        return JudgeResult(status="uncertain", reason=temporal_warning)
            # Table integrity check — malformed tables degrade response quality
            table_issue = _check_table_integrity(answer)
            if table_issue:
                logger.warning("Judge table integrity: %s", table_issue)
                return JudgeResult(status="uncertain", reason=table_issue)
            # Answer relevance check (intent-aware threshold)
            relevance = _check_answer_relevance(query, answer, intent)
            threshold = _get_relevance_threshold(intent)
            if relevance < threshold:
                return JudgeResult(status="fail", reason=f"answer_irrelevant (relevance={relevance}, threshold={threshold})")
            # Response structure quality check
            structure_issue = _check_response_structure(answer, intent, query=query)
            if structure_issue:
                logger.warning("Judge structure check: %s", structure_issue)
                return JudgeResult(status="uncertain", reason=structure_issue)
            # Answer-evidence factual consistency check
            if evidence_texts and not _is_self_evidence:
                consistency_issue = _check_factual_consistency(answer, evidence_texts)
                if consistency_issue:
                    logger.warning("Judge factual consistency: %s", consistency_issue)
                    return JudgeResult(status="uncertain", reason=consistency_issue)
            # Completeness check
            if evidence_texts and not _is_self_evidence:
                completeness_warning = _check_answer_completeness(query, answer, evidence_texts, intent)
                if completeness_warning:
                    logger.warning("Judge completeness check: %s", completeness_warning)
                    return JudgeResult(status="fail", reason=completeness_warning)

    if heuristic.status != "uncertain":
        # Compute graduated confidence for non-uncertain verdicts
        heuristic.confidence = _compute_judge_confidence(
            answer, schema, intent, query, heuristic.status
        )
        return heuristic

    if llm_client and budget.consume():
        llm_result = _llm_judge(answer, schema, intent, llm_client, correlation_id, query)
        if llm_result:
            llm_result.confidence = _compute_judge_confidence(
                answer, schema, intent, query, llm_result.status
            )
            return llm_result

    # If heuristic is uncertain but the schema has any usable content,
    # pass rather than killing a potentially valid response.
    if _has_any_content(schema):
        return JudgeResult(status="pass", reason="uncertain_with_content", confidence=0.5)
    return JudgeResult(status="fail", reason="no_usable_content", confidence=0.1)

def _compute_judge_confidence(
    answer: str,
    schema: Any,
    intent: str,
    query: str,
    verdict: str,
) -> float:
    """Compute a graduated confidence score (0.0-1.0) for the judge verdict.

    Combines multiple signals:
    - Base score from verdict (pass=0.7, uncertain=0.4, fail=0.2)
    - Relevance boost/penalty
    - Entity coverage
    - Evidence density
    - Answer structure quality
    """
    # Base confidence from verdict
    if verdict == "pass":
        score = 0.70
    elif verdict == "uncertain":
        score = 0.40
    else:
        score = 0.15

    # Relevance signal
    if query and answer:
        relevance = _check_answer_relevance(query, answer, intent)
        if relevance >= 0.5:
            score += 0.10  # Good relevance bonus
        elif relevance < 0.2:
            score -= 0.10  # Poor relevance penalty

    # Evidence density: more evidence spans = more confidence
    try:
        spans = list(_iter_spans(schema))
        n_spans = len(spans)
        if n_spans >= 5:
            score += 0.08
        elif n_spans >= 2:
            score += 0.04
        elif n_spans == 0:
            score -= 0.10
    except Exception:
        pass

    # Answer length appropriateness
    answer_len = len(answer.strip())
    if 50 <= answer_len <= 2000:
        score += 0.05  # Reasonable length
    elif answer_len < 20:
        score -= 0.10  # Suspiciously short

    # Table/structure quality for structured intents
    if intent in ("comparison", "ranking", "analytics", "multi_field"):
        if "|" in answer and "---" in answer:
            score += 0.05  # Has table structure

    # Intent-aware adjustment: relaxed thresholds for generative/reasoning intents
    # (these naturally paraphrase evidence rather than quoting verbatim)
    _INTENT_CONFIDENCE_ADJUST = {
        "factual": 0.05,      # factual needs higher bar — exact values expected
        "contact": 0.05,      # contact info must be precise
        "reasoning": -0.05,   # reasoning paraphrases are expected
        "generate": -0.10,    # creative output diverges from evidence
        "summary": -0.03,     # summaries naturally compress
        "cross_document": -0.03,  # synthesis across docs uses paraphrasing
    }
    score += _INTENT_CONFIDENCE_ADJUST.get(intent, 0.0)

    # Self-consistency check: if answer contradicts itself, penalize
    if answer:
        _answer_lower = answer.lower()
        # Simple self-contradiction: same number appears with different values
        import re as _re_jc
        _nums = _re_jc.findall(r'\b(\d+(?:[.,]\d+)?)\s*(?:years?|months?|%|dollars?|\$)', _answer_lower)
        if len(_nums) != len(set(_nums)):
            # Duplicate numbers may be fine (repeated for emphasis), don't penalize
            pass

    return round(max(0.0, min(1.0, score)), 3)

def _has_any_content(
    schema: InvoiceSchema | HRSchema | LegalSchema | GenericSchema | MultiEntitySchema | LLMResponseSchema,
) -> bool:
    """Check if schema has any extractable content worth showing."""
    if isinstance(schema, LLMResponseSchema):
        return bool(schema.text and len(schema.text) > 20)
    if isinstance(schema, HRSchema):
        cands = (schema.candidates.items if schema.candidates else []) or []
        return any(getattr(c, "name", None) or getattr(c, "skills", None) or getattr(c, "experience", None) for c in cands)
    if isinstance(schema, GenericSchema):
        facts = (schema.facts.items if schema.facts else []) or []
        return any(f.value and len(str(f.value)) > 10 for f in facts)
    if isinstance(schema, InvoiceSchema):
        items = (schema.items.items if schema.items else []) or []
        return bool(items)
    # For any other schema, check if it has any evidence spans
    try:
        return any(True for _ in _iter_spans(schema))
    except Exception:
        return False

def _heuristic_judge(
    answer: str,
    schema: InvoiceSchema | HRSchema | LegalSchema | GenericSchema | MultiEntitySchema | LLMResponseSchema,
    intent: str,
    query: str = "",
) -> JudgeResult:
    if _has_forbidden_tokens(answer):
        return JudgeResult(status="fail", reason="forbidden_tokens")

    # LLMResponseSchema: the LLM already used document evidence to produce
    # the answer, so only check forbidden tokens + minimum length.
    if isinstance(schema, LLMResponseSchema):
        if len(answer.strip()) < 10:
            return JudgeResult(status="fail", reason="too_short")
        return JudgeResult(status="pass")

    if isinstance(schema, MultiEntitySchema):
        lowered = answer.lower()
        has_multi_signal = (
            "multiple" in lowered
            or bool(re.search(r"\d+\.\s", answer))  # numbered list
            or answer.count("\n- ") >= 2  # 2+ bullet lines
            or ("|" in answer and "---" in answer)  # markdown table
            or any(tok in lowered for tok in ("candidates", "documents", "resumes", "entries"))
        )
        if not has_multi_signal:
            return JudgeResult(status="fail", reason="multi_entity_not_explicit")

    if isinstance(schema, InvoiceSchema) and intent == "products_list":
        items = schema.items.items if schema.items else None
        if not items:
            if "itemized products/services" in answer:
                return JudgeResult(status="pass")
            if _extract_candidate_items(answer):
                return JudgeResult(status="fail", reason="items_without_evidence")
            return JudgeResult(status="pass")

        allowed_items = {_normalize(item.description) for item in items if item.description}
        for candidate in _extract_candidate_items(answer):
            if _normalize(candidate) not in allowed_items:
                return JudgeResult(status="fail", reason="hallucinated_item")

    has_spans = any(True for _ in _iter_spans(schema))
    if not has_spans:
        return JudgeResult(status="fail", reason="no_evidence_spans")

    # Evidence span quality gate: spans shorter than threshold are likely
    # fragments that don't provide meaningful provenance.  If ALL spans are
    # short, the extraction is unreliable.
    # Domain-aware: invoice/medical schemas legitimately have short values
    # (SKU codes, lab values "Positive", amounts "$1500"), so use a lower threshold.
    from .types import MedicalSchema, PolicySchema
    if isinstance(schema, (InvoiceSchema, GenericSchema, MedicalSchema, PolicySchema)):
        _short_threshold = 5  # Allow very short values for structured domains
    else:
        _short_threshold = 15
    _spans_list = list(_iter_spans(schema))
    if _spans_list:
        _short_spans = sum(1 for s in _spans_list if len(s.strip()) < _short_threshold)
        if _short_spans == len(_spans_list) and len(_spans_list) >= 2:
            return JudgeResult(status="uncertain", reason="evidence_spans_too_short")

    # HR schemas with valid candidates: numbers in the rendered ranking
    # (e.g. "16 years experience") come from deterministic extraction of
    # chunk text, NOT from evidence_spans.  Skip the hallucinated_number
    # check — the data is trustworthy when sourced from document content.
    _hr_has_valid_candidates = (
        isinstance(schema, HRSchema)
        and schema.candidates
        and getattr(schema.candidates, "items", None)
        and any(c.name for c in schema.candidates.items)
    )

    # GenericSchema with valid facts: numbers come from deterministic
    # extraction of chunk text (KV pairs, list items).  Same bypass logic.
    _generic_has_valid_facts = (
        isinstance(schema, GenericSchema)
        and schema.facts
        and getattr(schema.facts, "items", None)
        and any(f.label for f in schema.facts.items)
    )

    answer_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", answer))
    if answer_numbers and not _hr_has_valid_candidates and not _generic_has_valid_facts:
        evidence_numbers = _collect_evidence_numbers(schema)
        if not evidence_numbers:
            return JudgeResult(status="uncertain", reason="numbers_without_evidence")
        # Use the normalized checker (handles $50K vs 50,000, locale variants, etc.)
        # instead of raw string set comparison which causes false positives
        _evidence_spans = list(_iter_spans(schema))
        if _evidence_spans:
            hallucinated = _check_numeric_fidelity(answer, _evidence_spans)
            if len(hallucinated) >= 3:
                return JudgeResult(status="fail", reason="hallucinated_number")
        elif answer_numbers - evidence_numbers:
            # Fallback to raw comparison only when no evidence spans available
            return JudgeResult(status="fail", reason="hallucinated_number")

    # Completeness is checked in the main judge() path with proper severity
    # — removed duplicate heuristic check that returned weaker "uncertain"

    return JudgeResult(status="pass")

def _collect_evidence_numbers(schema: InvoiceSchema | HRSchema | LegalSchema | GenericSchema | MultiEntitySchema) -> set:
    numbers = set()
    for span in _iter_spans(schema):
        numbers.update(re.findall(r"\d+(?:[.,]\d+)?", span))
    return numbers

def _iter_spans(schema: InvoiceSchema | HRSchema | LegalSchema | GenericSchema | MultiEntitySchema | LLMResponseSchema) -> Iterable[str]:
    # LLMResponseSchema: the LLM answer text itself serves as evidence.
    if isinstance(schema, LLMResponseSchema):
        if schema.text:
            yield schema.text
        return
    if isinstance(schema, InvoiceSchema):
        items = schema.items.items if schema.items else None
        for item in items or []:
            for span in item.evidence_spans:
                yield span.snippet
        for group in (schema.totals, schema.parties, schema.terms):
            group_items = group.items if group else None
            for item in group_items or []:
                for span in item.evidence_spans:
                    yield span.snippet
    elif isinstance(schema, HRSchema):
        candidates = schema.candidates.items if schema.candidates else None
        for cand in candidates or []:
            for span in cand.evidence_spans:
                yield span.snippet
    elif isinstance(schema, LegalSchema):
        clauses = schema.clauses.items if schema.clauses else None
        for clause in clauses or []:
            for span in clause.evidence_spans:
                yield span.snippet
    elif isinstance(schema, GenericSchema):
        facts = schema.facts.items if schema.facts else None
        for fact in facts or []:
            for span in fact.evidence_spans:
                yield span.snippet
    elif isinstance(schema, MultiEntitySchema):
        for entity in schema.entities or []:
            for span in entity.evidence_spans:
                yield span.snippet

def _extract_candidate_items(answer: str) -> list[str]:
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    candidates = []
    for line in lines:
        if line.startswith("-"):
            candidates.append(line.lstrip("- ").strip())
    if not candidates and "Items listed" in answer:
        after = answer.split(":", 1)[-1]
        for part in after.split(";"):
            part = part.strip(" .")
            if part:
                candidates.append(part)
    return candidates

def _normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()

_LLM_JUDGE_TIMEOUT_S = 15.0

def _llm_judge(
    answer: str,
    schema: InvoiceSchema | HRSchema | LegalSchema | GenericSchema | MultiEntitySchema,
    intent: str,
    llm_client: object,
    correlation_id: Optional[str],
    query: str = "",
) -> Optional[JudgeResult]:
    prompt = _build_prompt(answer, schema, intent, query)

    def _call() -> Optional[str]:
        return llm_client.generate(prompt, max_retries=1, backoff=0.4)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_call)
            raw = future.result(timeout=_LLM_JUDGE_TIMEOUT_S)
    except concurrent.futures.TimeoutError:
        logger.warning(
            "RAG v3 LLM judge timed out after %.1fs",
            _LLM_JUDGE_TIMEOUT_S,
            extra={"stage": "judge", "correlation_id": correlation_id},
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "RAG v3 LLM judge failed: %s",
            exc,
            extra={"stage": "judge", "correlation_id": correlation_id},
        )
        return None

    payload = _extract_json(raw)
    verdict = (payload.get("verdict") or "").lower() if isinstance(payload, dict) else ""
    if verdict in {"pass", "fail"}:
        return JudgeResult(status=verdict, reason=str(payload.get("reason") or ""))
    return None

def _infer_domain_label(
    schema: InvoiceSchema | HRSchema | LegalSchema | GenericSchema | MultiEntitySchema,
) -> str:
    """Infer a human-readable domain label from the schema type."""
    if isinstance(schema, InvoiceSchema):
        return "invoice/financial"
    if isinstance(schema, HRSchema):
        return "HR/recruitment"
    if isinstance(schema, LegalSchema):
        return "legal/compliance"
    from .types import MedicalSchema, PolicySchema
    if isinstance(schema, MedicalSchema):
        return "medical/clinical"
    if isinstance(schema, PolicySchema):
        return "policy/regulatory"
    if isinstance(schema, MultiEntitySchema):
        return "multi-document"
    return "general"

def _build_prompt(
    answer: str,
    schema: InvoiceSchema | HRSchema | LegalSchema | GenericSchema | MultiEntitySchema,
    intent: str,
    query: str = "",
) -> str:
    evidence = []
    for span in _iter_spans(schema):
        snippet = " ".join(span.split())[:200]
        if snippet:
            evidence.append(f"- {snippet}")

    domain = _infer_domain_label(schema)

    # Build structured evaluation prompt with criteria and few-shot example
    prompt_parts = [
        "You are a strict answer-quality judge for a document intelligence system.",
        f"Domain: {domain}. Intent: {intent}.",
        "",
        "## Evaluation Criteria",
        "1. **Grounding**: Every factual claim in the answer MUST be supported by the evidence snippets.",
        "2. **Relevance**: The answer must directly address the question asked.",
        "3. **Completeness**: The answer should cover the key points from evidence (not omit critical facts).",
        "4. **No Hallucination**: Numbers, names, dates, and specific claims must appear in evidence.",
        "5. **No LLM Artifacts**: No phrases like 'based on the provided context' or 'as an AI'.",
        "",
        "## Output Format",
        'Return ONLY strict JSON: {"verdict": "pass"|"fail", "reason": "<brief explanation>"}',
        "",
        "## Example",
        'QUERY: What is the total invoice amount?',
        'ANSWER: The total invoice amount is $5,250.00.',
        'EVIDENCE:',
        '- Invoice Total: $5,250.00 | Due Date: 2024-03-15',
        'JSON: {"verdict": "pass", "reason": "Total amount $5,250.00 directly supported by evidence."}',
        "",
        '---',
        'QUERY: What is the total invoice amount?',
        'ANSWER: The total invoice amount is approximately $6,000.',
        'EVIDENCE:',
        '- Invoice Total: $5,250.00 | Due Date: 2024-03-15',
        'JSON: {"verdict": "fail", "reason": "Answer states $6,000 but evidence shows $5,250.00."}',
        "",
        "## Your Evaluation",
    ]

    if query:
        prompt_parts.append(f"QUERY: {query}")

    prompt_parts.append(f"ANSWER: {answer}")
    prompt_parts.append("EVIDENCE:")
    prompt_parts.extend(evidence if evidence else ["- (no evidence snippets)"])
    prompt_parts.append("")
    prompt_parts.append("JSON:")

    return "\n".join(prompt_parts)

def _extract_json(raw: object) -> dict:
    if not raw:
        return {}
    text = str(raw).strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            return {}
    if "{" in text and "}" in text:
        snippet = text[text.find("{") : text.rfind("}") + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return {}
    return {}

def _check_temporal_consistency(
    answer: str, evidence_texts: list[str],
) -> Optional[str]:
    """Check that dates in the answer are temporally consistent with evidence.

    Detects: future dates presented as past, chronological inversions,
    and dates not found in evidence at all.

    Returns a warning string if inconsistency detected, None if OK.
    """
    _DATE_EXTRACT_RE = re.compile(
        r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b"
        r"|\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b"
        r"|\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)\s+(\d{1,2}),?\s+(\d{4})\b"
    )
    _MONTH_MAP = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }

    def _extract_year_month(text: str) -> list[tuple[int, int]]:
        dates = []
        for m in _DATE_EXTRACT_RE.finditer(text):
            if m.group(1):  # ISO: YYYY-MM-DD
                dates.append((int(m.group(1)), int(m.group(2))))
            elif m.group(4):  # US: MM/DD/YYYY
                dates.append((int(m.group(6)), int(m.group(4))))
            elif m.group(7):  # Named: March 15, 2024
                month_num = _MONTH_MAP.get(m.group(7)[:3].lower(), 0)
                if month_num:
                    dates.append((int(m.group(9)), month_num))
        # Also extract standalone years: "2019 - 2023", "since 2020"
        for ym in re.finditer(r"\b(20[12]\d)\b", text):
            yr = int(ym.group(1))
            if not any(d[0] == yr for d in dates):
                dates.append((yr, 0))
        return dates

    answer_dates = _extract_year_month(answer)
    if not answer_dates:
        return None

    evidence_combined = " ".join(evidence_texts)
    evidence_dates = _extract_year_month(evidence_combined)
    evidence_years = {d[0] for d in evidence_dates}

    # Check for years in answer not found in evidence
    answer_years = {d[0] for d in answer_dates}
    ungrounded_years = answer_years - evidence_years
    # Filter trivial/common years
    ungrounded_years = {y for y in ungrounded_years if y >= 2000}

    # Allow continuous year ranges: if ungrounded years form a range bridging
    # evidence years (e.g., answer has 2018-2023, evidence has 2018 and 2023),
    # the intermediate years are contextually grounded
    if ungrounded_years and evidence_years:
        ev_min, ev_max = min(evidence_years), max(evidence_years)
        ungrounded_years = {y for y in ungrounded_years if y < ev_min - 1 or y > ev_max + 1}

    if len(ungrounded_years) >= 3:
        return f"temporal_ungrounded: years {ungrounded_years} not in evidence"

    return None

def _check_response_structure(answer: str, intent: str, query: str = "") -> Optional[str]:
    """Check for structural quality issues in the response.

    Detects: truncated tables, incomplete lists, empty sections,
    and other formatting issues that degrade readability.
    Returns a warning string if issues found, None if OK.
    """
    if not answer or len(answer) < 50:
        return None

    lines = answer.strip().splitlines()

    # Check for truncated table: table starts but no data rows
    in_table = False
    table_data_rows = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|"):
            # Separator row: contains only pipes, dashes, colons, spaces
            if re.match(r"^[\s\-:|]+$", stripped):
                continue  # separator
            in_table = True
            table_data_rows += 1
        elif in_table and not stripped.startswith("|"):
            break
    if in_table and table_data_rows <= 1:
        # Only header row, no data — truncated table
        return "truncated_table: table has header but no data rows"

    # Check for empty bold sections: **Header:** followed by nothing
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^\*\*[^*]+\*\*:?\s*$", stripped):
            # Bold header on its own line — check if next line has content
            next_lines = [l.strip() for l in lines[i+1:i+3] if l.strip()]
            if not next_lines:
                return "empty_section: section header with no content following"

    # Check for incomplete numbered lists for multi-entity intents
    numbered_items = [l.strip() for l in lines if re.match(r"^\d+[.)]\s", l.strip())]
    if intent in ("ranking", "comparison", "cross_document"):
        if len(numbered_items) == 1:
            return "incomplete_list: only 1 numbered item for multi-entity intent"
        # Also check bullet/bold items for comparison intents
        bold_items = [l for l in lines if re.match(r"^\s*[-*]\s+\*\*", l)]
        list_count = max(len(numbered_items), len(bold_items))
        if 0 < list_count < 2:
            return "incomplete_list: fewer than 2 items for multi-entity intent"

    # Repeated content detection: same paragraph appearing twice
    paragraphs = re.split(r"\n\s*\n", answer)
    if len(paragraphs) >= 3:
        _para_keys = [re.sub(r"\s+", " ", p.strip().lower()) for p in paragraphs if p.strip()]
        _para_set = set(_para_keys)
        if len(_para_set) < len(_para_keys) * 0.7:
            return "repeated_content: same paragraph appears multiple times"

    # Truncated list item detection: numbered item with no content after number
    for line in lines:
        stripped = line.strip()
        if re.match(r"^\d+[.)]\s*$", stripped):
            return "truncated_list_item: numbered item with no content"

    # Table column consistency: rows must have the same number of columns
    if in_table and table_data_rows >= 2:
        _table_col_counts = []
        _in_tbl = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                if not re.match(r"^[\s\-:|]+$", stripped):
                    _cols = [c for c in stripped.strip("|").split("|")]
                    _table_col_counts.append(len(_cols))
                _in_tbl = True
            elif _in_tbl:
                break
        if _table_col_counts and len(set(_table_col_counts)) > 1:
            return "inconsistent_table: rows have different column counts"

    # Wall-of-text detection: long responses (>500 chars) with no structure
    # (no bullets, tables, headers, or line breaks) are hard to read
    _STRUCTURE_INTENTS = frozenset({
        "comparison", "ranking", "cross_document", "analytics", "multi_field",
        "timeline", "summary",
    })
    if intent in _STRUCTURE_INTENTS and len(answer) > 500:
        has_structure = (
            "|" in answer  # table
            or "- " in answer  # bullets
            or "* " in answer  # bullets
            or re.search(r"^\d+[.)]\s", answer, re.MULTILINE)  # numbered list
            or re.search(r"^#{1,4}\s", answer, re.MULTILINE)  # headers
            or answer.count("\n") >= 3  # multi-paragraph
        )
        if not has_structure:
            return "wall_of_text: long structured-intent response with no formatting"

    # Format-mismatch detection: query explicitly asks for a table but response has none
    _ql = query.lower() if query else ""
    _table_requested = any(w in _ql for w in ("table", "tabular", "spreadsheet", "grid format"))
    if _table_requested and "|" not in answer and len(answer) > 100:
        return "format_mismatch: query requested table format but response contains no table"

    # Empty/mostly-N/A table detection: table exists but >70% data cells are N/A or empty
    if in_table and table_data_rows >= 2:
        _na_cells = 0
        _total_cells = 0
        _in_data_tbl = False
        _header_seen = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                if re.match(r"^[\s\-:|]+$", stripped):
                    _header_seen = True
                    continue  # separator
                if not _header_seen:
                    _in_data_tbl = True
                    continue  # header row
                # Data row
                _in_data_tbl = True
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                for cell in cells:
                    _total_cells += 1
                    if cell.lower() in ("n/a", "na", "-", "—", "–", "none", "not found", ""):
                        _na_cells += 1
            elif _in_data_tbl:
                break
        if _total_cells >= 4 and _na_cells / _total_cells > 0.70:
            return "empty_table: over 70% of table data cells are N/A or empty"

    return None

def _has_forbidden_tokens(answer: str) -> bool:
    lowered = answer.lower()
    for token in (
        "understanding & scope",
        "evidence & gaps",
        "invoice summary",
        "files searched",
        "files used",
        "documents searched",
        "sources used",
        # LLM meta-commentary artifacts
        "based on the provided",
        "based on the context provided",
        "based on the given",
        "i don't have access",
        "i cannot access",
        "as an ai",
        "as a language model",
        "i'm unable to",
        "i am unable to",
        "let me search",
        "let me look",
        "i'll analyze",
        "i will analyze",
        "please note that i",
        "as a helpful assistant",
        "the provided documents show",
        "the documents provided",
        "here is my analysis",
        "here's my analysis",
        # Additional GPT-style artifacts
        "upon review",
        "upon reviewing",
        "upon examination",
        "it appears that",
        "it seems that",
        "it is worth noting",
        "it's worth noting",
        "it should be noted",
        "it is important to note",
        "in conclusion, based on",
        "from the provided information",
        "from the given information",
        "after careful review",
        "after careful analysis",
    ):
        if token in lowered:
            return True
    if re.search(r"^\s*answer\s*[:\-]", lowered, re.MULTILINE):
        return True
    return False

def judge(
    *,
    answer: str,
    schema: InvoiceSchema | HRSchema | LegalSchema | GenericSchema | MultiEntitySchema,
    intent: str,
    llm_client: Optional[object],
    budget: LLMBudget,
    sources_present: bool = True,
    correlation_id: Optional[str] = None,
    query: str = "",
) -> JudgeResult:
    return judge_answer(
        answer=answer,
        schema=schema,
        intent=intent,
        llm_client=llm_client,
        budget=budget,
        sources_present=sources_present,
        correlation_id=correlation_id,
        query=query,
    )
