"""
Hallucination self-correction for DocWain.

When grounding gate detects low confidence, automatically re-generates with
tighter constraints. Uses sentence-level Jaccard grounding (reuses fast_grounding
pattern) and either LLM correction or deterministic removal of unsupported sentences.

Enhanced with synonym-aware scoring, domain-specific negation detection,
confidence-weighted correction priority, and missing-evidence detection.
"""

from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_NUMBER_RE = re.compile(r"\b\d[\d,.]*\b")
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")

# ── Negation tokens (expanded per domain) ──────────────────────────────────

_NEGATION_TOKENS: FrozenSet[str] = frozenset({
    # General
    "not", "no", "never", "none", "neither", "nor", "without",
    "absent", "negative", "denied", "denies", "unremarkable",
    # Medical
    "contraindicated", "benign", "afebrile", "asymptomatic",
    "no_evidence_of",
    # Legal
    "unenforceable", "void", "invalid", "waived", "inapplicable",
    # Financial
    "unpaid", "overdue", "outstanding", "disputed", "declined",
    # HR
    "unqualified", "inexperienced", "lacking", "insufficient",
})

# Tokens that are negation-like only in specific domains
_DOMAIN_NEGATION_TOKENS: Dict[str, FrozenSet[str]] = {
    "medical": frozenset({
        "contraindicated", "unremarkable", "benign", "non-reactive",
        "afebrile", "asymptomatic", "no_evidence_of",
    }),
    "legal": frozenset({
        "unenforceable", "void", "invalid", "waived", "inapplicable",
    }),
    "invoice": frozenset({
        "unpaid", "overdue", "outstanding", "disputed", "declined",
    }),
    "hr": frozenset({
        "unqualified", "inexperienced", "lacking", "insufficient",
    }),
}

# ── Synonym groups for semantic scoring ─────────────────────────────────────

_SYNONYM_GROUPS: List[FrozenSet[str]] = [
    frozenset({"salary", "compensation", "pay", "wage", "remuneration", "stipend"}),
    frozenset({"experience", "background", "history", "tenure"}),
    frozenset({"skills", "expertise", "proficiency", "competency", "abilities"}),
    frozenset({"medication", "drug", "prescription", "medicine", "pharmaceutical"}),
    frozenset({"diagnosis", "condition", "finding", "assessment", "evaluation"}),
    frozenset({"clause", "provision", "section", "article", "paragraph"}),
    frozenset({"payment", "remittance", "amount", "disbursement", "transaction"}),
    frozenset({"coverage", "benefit", "protection", "entitlement", "allowance"}),
    frozenset({"employee", "worker", "staff", "personnel", "associate"}),
    frozenset({"patient", "individual", "subject", "client", "applicant"}),
    frozenset({"contract", "agreement", "arrangement", "pact"}),
    frozenset({"liability", "obligation", "responsibility", "duty"}),
    frozenset({"treatment", "therapy", "intervention", "regimen", "procedure"}),
    frozenset({"qualification", "credential", "certification", "accreditation"}),
    frozenset({"university", "college", "institution", "school", "academy"}),
    frozenset({"deadline", "due_date", "expiry", "expiration", "maturity"}),
    frozenset({"address", "location", "residence", "domicile"}),
    frozenset({"manager", "supervisor", "director", "lead", "head"}),
    frozenset({"company", "organization", "firm", "corporation", "enterprise"}),
    frozenset({"revenue", "income", "earnings", "proceeds", "turnover"}),
]

# Build fast lookup: token -> set of synonyms (including itself)
_SYNONYM_MAP: Dict[str, FrozenSet[str]] = {}
for _group in _SYNONYM_GROUPS:
    for _word in _group:
        _SYNONYM_MAP[_word] = _group

# ── Domain-specific Jaccard thresholds ──────────────────────────────────────

_DOMAIN_THRESHOLDS: Dict[str, float] = {
    "medical": 0.55,   # Stricter — medication/diagnosis errors are dangerous
    "legal": 0.40,     # More lenient — boilerplate paraphrasing is normal
    "invoice": 0.45,   # Moderate — numbers matter but formatting varies
    "hr": 0.50,        # Default
    "policy": 0.45,    # Similar to invoice
}

# ── Correction prompts ──────────────────────────────────────────────────────

_CORRECTION_PROMPT = (
    "Rewrite the following sentence using ONLY facts from the evidence below. "
    "Preserve exact numbers, names, and dates from the evidence. "
    "If the claim cannot be supported by the evidence, respond with exactly: REMOVE\n\n"
    "Example:\n"
    "Sentence: The patient was prescribed 500mg of Amoxicillin daily.\n"
    "Evidence: Patient prescribed Amoxicillin 250mg twice daily for 10 days.\n"
    "Rewritten sentence: The patient was prescribed Amoxicillin 250mg twice daily for 10 days.\n\n"
    "Now rewrite this:\n"
    "Sentence: {sentence}\n\n"
    "Evidence:\n{evidence}\n\n"
    "Rewritten sentence:"
)

_BATCH_CORRECTION_PROMPT = (
    "For each numbered sentence below, rewrite it using ONLY facts from the evidence. "
    "Preserve exact numbers, names, and dates. Do not invent or infer facts. "
    "If a sentence cannot be supported by the evidence, respond with REMOVE for that sentence.\n"
    "Return one line per sentence in the format: NUMBER: rewritten sentence (or REMOVE)\n\n"
    "Example:\n"
    "1: The contract expires on March 2025.\n"
    "Evidence: Agreement termination date is June 30, 2026.\n"
    "1: The contract expires on June 30, 2026.\n\n"
    "Now correct these:\n"
    "SENTENCES:\n{sentences}\n\n"
    "EVIDENCE:\n{evidence}\n\n"
    "CORRECTIONS:"
)

# ── Correction priority levels ──────────────────────────────────────────────

_PRIORITY_NUMERIC = 3    # Sentences with numbers — highest risk
_PRIORITY_ENTITY = 2     # Sentences with named entities
_PRIORITY_GENERIC = 1    # Generic unsupported sentences

# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class MissingEvidence:
    """Evidence entity/fact present in chunks but absent from response."""
    text: str
    occurrences_in_evidence: int
    evidence_type: str  # "entity" or "number"

@dataclass
class CorrectionResult:
    """Result of hallucination correction pass."""

    original: str
    corrected: str
    corrections_made: int
    removed_sentences: List[str]
    corrected_sentences: List[Tuple[str, str]]  # (original, corrected)
    was_modified: bool
    missing_evidence: List[MissingEvidence] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "corrections_made": self.corrections_made,
            "removed_count": len(self.removed_sentences),
            "corrected_count": len(self.corrected_sentences),
            "was_modified": self.was_modified,
            "missing_evidence_count": len(self.missing_evidence),
        }

# ── Helpers ─────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> Set[str]:
    """Extract lowercase tokens from text."""
    return {t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 2}

def _expand_with_synonyms(tokens: Set[str]) -> Set[str]:
    """Expand a token set with known synonyms."""
    expanded = set(tokens)
    for tok in tokens:
        synonyms = _SYNONYM_MAP.get(tok)
        if synonyms:
            expanded.update(synonyms)
    return expanded

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, keeping table rows, list items, and headers intact."""
    lines = text.strip().splitlines()
    sentences: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or len(stripped) <= 5:
            continue
        # Keep table rows as single units (don't split on periods inside tables)
        if stripped.startswith("|") and stripped.endswith("|"):
            sentences.append(stripped)
            continue
        # Keep markdown headers as single units
        if stripped.startswith("#"):
            sentences.append(stripped)
            continue
        # Keep list items as single units
        if re.match(r"^\s*(?:[-*•]\s|\d+[.)]\s)", stripped):
            sentences.append(stripped)
            continue
        # Keep table separator rows
        if re.match(r"^\|[\s\-:]+\|", stripped):
            sentences.append(stripped)
            continue
        # Standard sentence split for prose
        parts = _SENTENCE_SPLIT_RE.split(stripped)
        sentences.extend(s.strip() for s in parts if s and len(s.strip()) > 5)
    return sentences

def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / max(len(union), 1)

def _get_negation_tokens(domain: Optional[str] = None) -> FrozenSet[str]:
    """Get negation tokens, optionally enriched with domain-specific ones."""
    if domain and domain.lower() in _DOMAIN_NEGATION_TOKENS:
        return _NEGATION_TOKENS | _DOMAIN_NEGATION_TOKENS[domain.lower()]
    return _NEGATION_TOKENS

def _has_negation_mismatch(
    sentence: str,
    evidence_texts: List[str],
    domain: Optional[str] = None,
) -> bool:
    """Detect if sentence has inverted meaning compared to evidence.

    E.g., "Patient does NOT have diabetes" vs evidence "Patient has diabetes"
    returns True (negation mismatch).
    """
    negation_set = _get_negation_tokens(domain)
    sent_lower = sentence.lower()
    sent_tokens = set(sent_lower.split())
    sent_has_negation = bool(sent_tokens & negation_set)

    if not sent_has_negation:
        return False

    # Check if evidence has the same content words WITHOUT negation
    content_words = {w for w in _TOKEN_RE.findall(sent_lower)
                     if len(w) > 2 and w not in negation_set}
    if len(content_words) < 2:
        return False

    for chunk in evidence_texts:
        chunk_lower = chunk.lower()
        chunk_tokens = set(chunk_lower.split())
        chunk_has_negation = bool(chunk_tokens & negation_set)

        # Check if same content words appear but with opposite negation
        chunk_content = {w for w in _TOKEN_RE.findall(chunk_lower)
                         if len(w) > 2 and w not in negation_set}
        overlap = content_words & chunk_content
        if len(overlap) >= len(content_words) * 0.6:
            # High content overlap — check if negation differs
            if sent_has_negation != chunk_has_negation:
                return True  # Negation mismatch detected

    return False

def _asymmetric_containment(response_tokens: Set[str], evidence_tokens: Set[str]) -> float:
    """What fraction of response tokens appear in evidence (asymmetric)."""
    if not response_tokens:
        return 0.0
    return len(response_tokens & evidence_tokens) / len(response_tokens)

def _score_sentence(
    sentence: str,
    chunk_texts: List[str],
    domain: Optional[str] = None,
) -> float:
    """Score a single sentence against all evidence chunks.

    Uses both Jaccard and asymmetric containment, with synonym expansion
    and negation mismatch penalty.

    When the sentence contains named entities, prioritizes evidence from
    chunks that mention those same entities to avoid cross-document
    hallucination (e.g., attributing one candidate's skills to another).
    """
    sent_tokens = _tokenize(sentence)
    if not sent_tokens:
        return 1.0  # Empty/trivial sentences pass

    # Entity-aware evidence selection: if sentence mentions specific entities,
    # restrict primary scoring to chunks containing those entities.
    # This prevents cross-document hallucinations where "John's 8 years"
    # scores as supported because "8 years" appears in Jane's resume.
    _sent_entities = set(_ENTITY_RE.findall(sentence))
    _entity_filtered_texts = chunk_texts  # default: all chunks
    if _sent_entities and len(chunk_texts) > 1:
        _entity_lower = {e.lower() for e in _sent_entities if len(e) > 2}
        if _entity_lower:
            _matching = [c for c in chunk_texts if any(e in c.lower() for e in _entity_lower)]
            if _matching:
                _entity_filtered_texts = _matching

    all_evidence_tokens: Set[str] = set()
    for chunk in _entity_filtered_texts:
        all_evidence_tokens.update(_tokenize(chunk))

    # Standard Jaccard
    base_score = _jaccard(sent_tokens, all_evidence_tokens)

    # Asymmetric containment — what fraction of sentence tokens are in evidence
    # This is more useful than Jaccard when evidence is much larger than sentence
    containment_score = _asymmetric_containment(sent_tokens, all_evidence_tokens)

    # Synonym-expanded scores
    expanded_sent = _expand_with_synonyms(sent_tokens)
    expanded_evidence = _expand_with_synonyms(all_evidence_tokens)
    synonym_jaccard = _jaccard(expanded_sent, expanded_evidence)
    synonym_containment = _asymmetric_containment(expanded_sent, expanded_evidence)

    # Stem-like matching: count sentence tokens that share a common prefix (≥5 chars)
    # with evidence tokens — catches plural/tense variants without a full stemmer
    _unmatched = sent_tokens - all_evidence_tokens
    _stem_hits = 0
    if _unmatched:
        for _ut in _unmatched:
            if len(_ut) >= 5:
                _prefix = _ut[:max(5, len(_ut) - 3)]
                if any(et[:len(_prefix)] == _prefix for et in all_evidence_tokens if len(et) >= len(_prefix)):
                    _stem_hits += 1
    # Compute stem-enhanced containment
    _effective_overlap = len(sent_tokens & all_evidence_tokens) + _stem_hits
    stem_containment = _effective_overlap / len(sent_tokens) if sent_tokens else 0.0

    # Take the best of all scoring methods
    score = max(base_score, containment_score, synonym_jaccard, synonym_containment, stem_containment)

    # If entity-filtered scoring was used and score is low, try full pool as fallback
    # (handles cases where entity appears only in a different chunk than the fact)
    if _entity_filtered_texts is not chunk_texts and score < 0.3:
        _all_tokens: Set[str] = set()
        for chunk in chunk_texts:
            _all_tokens.update(_tokenize(chunk))
        _full_containment = _asymmetric_containment(sent_tokens, _all_tokens)
        # Use full-pool score but discount it (cross-doc evidence is weaker signal)
        score = max(score, _full_containment * 0.75)

    # Penalize negation mismatches — sentence says opposite of evidence
    if _has_negation_mismatch(sentence, chunk_texts, domain=domain):
        score *= 0.3  # Heavy penalty for inverted meaning

    return score

def _classify_sentence_priority(sentence: str) -> int:
    """Classify correction priority for an unsupported sentence.

    Higher = more important to correct (higher risk of harmful hallucination).
    """
    has_numbers = bool(_NUMBER_RE.search(sentence))
    has_entities = bool(_ENTITY_RE.search(sentence))

    if has_numbers:
        return _PRIORITY_NUMERIC
    if has_entities:
        return _PRIORITY_ENTITY
    return _PRIORITY_GENERIC

def _detect_missing_evidence(
    response: str,
    chunk_texts: List[str],
    min_evidence_occurrences: int = 3,
) -> List[MissingEvidence]:
    """Detect critical evidence entities/numbers present in chunks but absent from response.

    Looks for entities and numbers that appear frequently in evidence (suggesting
    importance) but are not reflected in the generated response at all.

    Args:
        response: The generated response text
        chunk_texts: Evidence chunks
        min_evidence_occurrences: Minimum times an entity/number must appear in
            evidence to be considered important (default 3)

    Returns:
        List of MissingEvidence items for downstream use
    """
    if not response or not chunk_texts:
        return []

    response_lower = response.lower()
    combined_evidence = " ".join(chunk_texts)
    missing: List[MissingEvidence] = []

    # --- Detect missing entities ---
    entity_counter: Counter = Counter()
    for chunk in chunk_texts:
        # Extract capitalized multi-word entities
        for match in _ENTITY_RE.finditer(chunk):
            entity = match.group()
            # Skip very short or common words
            if len(entity) > 2 and entity.lower() not in {
                "the", "this", "that", "with", "from", "have", "been",
                "will", "would", "could", "should", "also", "such",
                "these", "those", "other", "some", "each", "more",
            }:
                entity_counter[entity] += 1

    for entity, count in entity_counter.items():
        if count >= min_evidence_occurrences:
            if entity.lower() not in response_lower:
                missing.append(MissingEvidence(
                    text=entity,
                    occurrences_in_evidence=count,
                    evidence_type="entity",
                ))

    # --- Detect missing numbers ---
    number_counter: Counter = Counter()
    for chunk in chunk_texts:
        for match in _NUMBER_RE.finditer(chunk):
            num = match.group()
            # Skip trivial numbers
            if num in {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}:
                continue
            number_counter[num] += 1

    for num, count in number_counter.items():
        if count >= min_evidence_occurrences:
            if num not in response:
                missing.append(MissingEvidence(
                    text=num,
                    occurrences_in_evidence=count,
                    evidence_type="number",
                ))

    if missing:
        logger.debug(
            "Missing evidence detected: %d items (entities: %d, numbers: %d)",
            len(missing),
            sum(1 for m in missing if m.evidence_type == "entity"),
            sum(1 for m in missing if m.evidence_type == "number"),
        )

    return missing

_STRUCTURAL_RE = re.compile(
    r"^\s*(?:"
    r"\*\*|##|###|####"              # Markdown headers/bold
    r"|\d+\.\s"                       # Numbered list items
    r"|[-*•]\s"                       # Bullet list items
    r"|[A-Z]\.\s"                     # Lettered list items (A. B. C.)
    r"|#+\s"                          # ATX headers
    r"|>{1,3}\s"                      # Blockquotes
    r"|based on|according to|from the" # Attribution intros
    r")"
)

def _is_structural(sentence: str) -> bool:
    """Check if sentence is structural (heading, list intro, attribution) and should be kept."""
    stripped = sentence.strip().rstrip(":")
    if len(stripped) < 20:
        return True
    if _STRUCTURAL_RE.match(stripped):
        return True
    if stripped.endswith(":"):
        return True
    # Table rows should always be kept
    if stripped.startswith("|") and stripped.endswith("|"):
        return True
    return False

# ── LLM correction helpers ──────────────────────────────────────────────────

def _call_llm_correction(
    llm_client: Any, sentence: str, evidence: str, timeout: float = 5.0
) -> Optional[str]:
    """Call LLM for single sentence correction with timeout."""
    if llm_client is None:
        return None

    prompt = _CORRECTION_PROMPT.format(sentence=sentence, evidence=evidence[:2000])

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_call_llm, llm_client, prompt)
            result = future.result(timeout=timeout)

        if not result:
            return None

        result = result.strip()
        if result.upper() == "REMOVE":
            return "REMOVE"
        if len(result) < 5:
            return None
        return result

    except (FuturesTimeout, Exception) as exc:
        logger.debug("LLM correction failed for sentence: %s", exc)
        return None

def _call_llm(llm_client: Any, prompt: str) -> str:
    """Call LLM client."""
    if hasattr(llm_client, "generate"):
        result = llm_client.generate(prompt)
        if isinstance(result, tuple):
            return result[0] if result[0] else ""
        return result or ""
    return ""

def _call_llm_batch_correction(
    llm_client: Any,
    sentences: List[Tuple[int, str]],
    evidence: str,
    timeout: float = 10.0,
) -> Dict[int, str]:
    """Batch-correct multiple sentences in a single LLM call.

    Args:
        llm_client: LLM client with generate() method
        sentences: List of (index, sentence) tuples to correct
        evidence: Evidence text for grounding
        timeout: Timeout for the single LLM call

    Returns:
        Dict mapping sentence index to correction result ("REMOVE" or rewritten text)
    """
    if llm_client is None or not sentences:
        return {}

    numbered = "\n".join(f"{idx}: {sent}" for idx, sent in sentences)
    prompt = _BATCH_CORRECTION_PROMPT.format(
        sentences=numbered, evidence=evidence[:2000],
    )

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_call_llm, llm_client, prompt)
            result = future.result(timeout=timeout)
    except (FuturesTimeout, Exception) as exc:
        logger.debug("Batch LLM correction failed: %s", exc)
        return {}

    if not result:
        return {}

    corrections: Dict[int, str] = {}
    # Try multiple delimiter patterns LLMs commonly use
    _DELIM_RE = re.compile(r"^(\d+)\s*[.:)]\s*(.*)")
    for line in result.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Parse "NUMBER: text", "NUMBER) text", "NUMBER. text" formats
        m = _DELIM_RE.match(line)
        if not m:
            continue
        try:
            idx = int(m.group(1))
        except ValueError:
            continue
        text = m.group(2).strip()
        if text.upper() == "REMOVE":
            corrections[idx] = "REMOVE"
        elif len(text) >= 5:
            corrections[idx] = text
    return corrections

# ── Main entry point ────────────────────────────────────────────────────────

def correct_hallucinations(
    response: str,
    chunk_texts: List[str],
    llm_client: Any = None,
    score_threshold: float = 0.5,
    max_corrections: int = 3,
    per_correction_timeout: float = 5.0,
    domain: Optional[str] = None,
) -> CorrectionResult:
    """
    Attempt to correct hallucinated sentences in the response.

    For each sentence below the score_threshold:
      1. Prioritize by severity (numeric > entity > generic)
      2. Try LLM correction with strict evidence prompt (budget goes to high-priority first)
      3. If LLM returns REMOVE or fails, deterministically remove the sentence
      4. Detect missing evidence facts for downstream use

    Args:
        response: The generated response text
        chunk_texts: Evidence chunks to ground against
        llm_client: Optional LLM client for re-generation
        score_threshold: Jaccard score below which a sentence is unsupported
        max_corrections: Maximum corrections to attempt
        per_correction_timeout: Timeout per LLM correction call
        domain: Optional domain hint for domain-adaptive thresholds

    Returns:
        CorrectionResult with corrected text and metadata
    """
    # Apply domain-specific threshold if available
    if domain and domain.lower() in _DOMAIN_THRESHOLDS:
        score_threshold = _DOMAIN_THRESHOLDS[domain.lower()]
    if not response or not chunk_texts:
        return CorrectionResult(
            original=response,
            corrected=response,
            corrections_made=0,
            removed_sentences=[],
            corrected_sentences=[],
            was_modified=False,
            missing_evidence=[],
        )

    sentences = _split_sentences(response)
    if not sentences:
        return CorrectionResult(
            original=response,
            corrected=response,
            corrections_made=0,
            removed_sentences=[],
            corrected_sentences=[],
            was_modified=False,
            missing_evidence=[],
        )

    evidence_text = "\n".join(chunk_texts[:10])
    removed: List[str] = []
    corrected_pairs: List[Tuple[str, str]] = []

    # Score all sentences and collect those needing correction
    unsupported: List[Tuple[int, str, float, int]] = []  # (index, sentence, score, priority)
    sentence_results: List[Optional[str]] = [None] * len(sentences)  # None = keep original

    prev_score = 0.0  # Track prior sentence grounding for coherence bonus
    for i, sentence in enumerate(sentences):
        if _is_structural(sentence):
            sentence_results[i] = sentence
            prev_score = 1.0  # Structural elements are always grounded
            continue

        score = _score_sentence(sentence, chunk_texts, domain=domain)

        # Paragraph coherence: if the previous sentence was well-grounded
        # (>=0.6), give this sentence a coherence bonus. This prevents
        # legitimate continuation sentences from being flagged as hallucinations.
        # E.g., "John has 8 years of Python." (grounded) followed by
        # "His expertise spans cloud architecture." (paraphrase, lower score)
        if prev_score >= 0.6 and score < score_threshold:
            coherence_bonus = min(0.15, (prev_score - 0.5) * 0.3)
            score = score + coherence_bonus

        prev_score = score

        if score >= score_threshold:
            sentence_results[i] = sentence
            continue

        priority = _classify_sentence_priority(sentence)
        unsupported.append((i, sentence, score, priority))

    # Sort by priority (highest first), then by score (lowest first — worst offenders)
    unsupported.sort(key=lambda x: (-x[3], x[2]))

    # Enforce correction budget — only top max_corrections get LLM attention
    budget_unsupported = unsupported[:max_corrections]
    over_budget = unsupported[max_corrections:]

    # Over-budget sentences: remove poorly-grounded ones without LLM correction.
    # Keep only sentences that score above half the threshold (moderate grounding).
    _keep_threshold = score_threshold * 0.5
    for idx, sentence, _score, priority in over_budget:
        if _score < _keep_threshold or (priority >= 2 and _score < 0.20):
            removed.append(sentence)
            logger.debug("Over-budget removal (score=%.2f, pri=%d): %s", _score, priority, sentence[:80])
        else:
            sentence_results[idx] = sentence

    # Batch-correct prioritized unsupported sentences in a single LLM call
    if budget_unsupported and llm_client is not None:
        batch_input = [(idx, sent) for idx, sent, _score, _prio in budget_unsupported]
        batch_results = _call_llm_batch_correction(
            llm_client, batch_input, evidence_text, timeout=per_correction_timeout * 2,
        )

        for idx, sentence, score, _prio in budget_unsupported:
            llm_result = batch_results.get(idx)
            if llm_result == "REMOVE":
                removed.append(sentence)
                logger.debug("Removed unsupported sentence: %s", sentence[:80])
            elif llm_result and llm_result != sentence:
                new_score = _score_sentence(llm_result, chunk_texts, domain=domain)
                if new_score > score:
                    sentence_results[idx] = llm_result
                    corrected_pairs.append((sentence, llm_result))
                    logger.debug(
                        "Corrected sentence (%.2f->%.2f): %s",
                        score, new_score, sentence[:80],
                    )
                else:
                    removed.append(sentence)
                    logger.debug(
                        "Correction didn't improve grounding, removing: %s",
                        sentence[:80],
                    )
            else:
                removed.append(sentence)
                logger.debug(
                    "No LLM correction available, removing unsupported: %s",
                    sentence[:80],
                )
    elif budget_unsupported:
        # No LLM client — remove unsupported sentences
        for idx, sentence, _score, _prio in budget_unsupported:
            removed.append(sentence)

    # Build final corrected parts from sentence_results
    corrected_parts: List[str] = []
    for result in sentence_results:
        if result is not None:
            corrected_parts.append(result)

    was_modified = bool(removed) or bool(corrected_pairs)
    corrected_text = "\n".join(corrected_parts).strip() if corrected_parts else response

    if corrected_text and len(corrected_text) < 20 and len(response) > 50:
        corrected_text = response
        was_modified = False

    # Detect missing evidence — important facts in chunks not reflected in response
    missing_evidence = _detect_missing_evidence(corrected_text, chunk_texts)

    return CorrectionResult(
        original=response,
        corrected=corrected_text,
        corrections_made=len(removed) + len(corrected_pairs),
        removed_sentences=removed,
        corrected_sentences=corrected_pairs,
        was_modified=was_modified,
        missing_evidence=missing_evidence,
    )
