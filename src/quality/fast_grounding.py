from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")  # CEO, IBM, HIPAA, etc.

# Table/list detection patterns
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s\-:]+\|\s*$")

_MODAL_VERBS = {
    "should",
    "must",
    "will",
    "may",
    "might",
    "can",
    "could",
    "would",
    "shall",
    "need",
    "needs",
    "required",
    "requires",
    "have to",
    "ought to",
    "need to",
    "must not",
    "better",
    "should not",
}

# --- Enhancement 1: expanded critical sentence markers ---
_TEMPORAL_RE = re.compile(
    r"\b(yesterday|tomorrow|Q[1-4]\s*\d{4}|next\s+(week|month|year)"
    r"|last\s+(week|month|year)|FY\s*\d{2,4})\b",
    re.IGNORECASE,
)
_COMPARATIVE_RE = re.compile(
    r"\b(higher|lower|greater|lesser|more than|less than|increased|decreased"
    r"|improved|declined|exceeded|below)\b",
    re.IGNORECASE,
)
_QUANTITATIVE_RE = re.compile(
    r"\b(all|every|none|no one|majority|most|few|several"
    r"|approximately|exactly|about|roughly|nearly)\b",
    re.IGNORECASE,
)

# --- Enhancement 2: negation-aware grounding ---
_NEGATION_TOKENS = frozenset({
    "not", "no", "never", "none", "absent", "negative", "denied",
    "without", "lacks", "missing", "excluded",
})
_NEGATION_PREFIX_RE = re.compile(
    r"\b(not|no|never|none|absent|negative|denied|without|lacks|missing|excluded"
    r"|does\s+not|did\s+not|do\s+not|has\s+not|have\s+not|is\s+not|was\s+not"
    r"|cannot|can\s*not|won't|wouldn't|shouldn't|couldn't|didn't|doesn't|hasn't"
    r"|haven't|isn't|wasn't|weren't)\b",
    re.IGNORECASE,
)

# --- Enhancement 5: semantic synonym pairs for key-term matching ---
_SYNONYM_PAIRS: list[tuple[str, str]] = [
    ("medication", "drug"),
    ("salary", "compensation"),
    ("candidate", "applicant"),
    ("diagnosis", "condition"),
    ("clause", "provision"),
    ("experience", "background"),
    ("treatment", "therapy"),
    ("obligation", "requirement"),
    ("premium", "cost"),
    ("vendor", "supplier"),
    ("employee", "worker"),
    ("certificate", "certification"),
    # Expanded pairs — aligned with judge.py synonym groups
    ("revenue", "income"),
    ("revenue", "earnings"),
    ("price", "cost"),
    ("price", "charge"),
    ("education", "degree"),
    ("education", "qualification"),
    ("skills", "competencies"),
    ("skills", "expertise"),
    ("company", "organization"),
    ("company", "firm"),
    ("deadline", "due date"),
    ("benefit", "allowance"),
    ("risk", "liability"),
    ("risk", "exposure"),
    ("contract", "agreement"),
    ("department", "division"),
    ("patient", "client"),
    ("address", "location"),
    ("phone", "telephone"),
    ("summary", "overview"),
    # Medical domain expansions
    ("adverse", "side effect"),
    ("dosage", "dose"),
    ("symptom", "complaint"),
    ("prognosis", "outlook"),
    ("prescription", "medication"),
    ("procedure", "surgery"),
    ("referral", "consultation"),
    # Legal domain expansions
    ("indemnification", "indemnity"),
    ("termination", "cancellation"),
    ("breach", "violation"),
    ("liability", "obligation"),
    ("arbitration", "dispute resolution"),
    ("warranty", "guarantee"),
    # Financial domain expansions
    ("expense", "cost"),
    ("discount", "reduction"),
    ("payment", "remittance"),
    ("invoice", "bill"),
    ("receivable", "outstanding"),
    ("margin", "profit"),
    # HR domain expansions
    ("staff", "employee"),
    ("tenure", "duration"),
    ("competency", "skill"),
    ("qualification", "credential"),
    ("role", "position"),
    ("promotion", "advancement"),
    ("attrition", "turnover"),
]
# Build a bidirectional lookup: word -> set of synonyms (lowercase)
_SYNONYM_MAP: dict[str, frozenset[str]] = {}
for _a, _b in _SYNONYM_PAIRS:
    _SYNONYM_MAP.setdefault(_a, set()).add(_b)  # type: ignore[arg-type]
    _SYNONYM_MAP.setdefault(_b, set()).add(_a)  # type: ignore[arg-type]
_SYNONYM_MAP = {k: frozenset(v) for k, v in _SYNONYM_MAP.items()}


@dataclass
class GroundingResult:
    supported_ratio: float
    critical_supported_ratio: float
    supported_sentences: List[str]
    unsupported_sentences: List[str]
    sentence_scores: List[float]


def _has_markdown_table(text: str) -> bool:
    """Detect if text contains a markdown table (pipes + separator row)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if _TABLE_LINE_RE.match(line) and i + 1 < len(lines):
            if _TABLE_SEP_RE.match(lines[i + 1]):
                return True
    return False


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    rough = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [s.strip() for s in rough if s.strip()]


def _is_table_or_list_line(line: str) -> bool:
    """Check if a line is part of a markdown table or list."""
    stripped = line.strip()
    if _TABLE_LINE_RE.match(stripped):
        return True
    if _TABLE_SEP_RE.match(stripped):
        return True
    # Bullet/numbered list items
    if re.match(r"^\s*[-*\u2022]\s+", stripped):
        return True
    if re.match(r"^\s*\d+[.)]\s+", stripped):
        return True
    return False


def _tokenize(text: str) -> List[str]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    return [t for t in tokens if len(t) > 2]


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a or not set_b:
        return 0.0
    inter = set_a & set_b
    union = set_a | set_b
    return len(inter) / max(len(union), 1)


def _containment(answer_tokens: Iterable[str], evidence_tokens: Iterable[str]) -> float:
    """Asymmetric containment: what fraction of answer tokens appear in evidence.

    Better than Jaccard for grounding because evidence is typically much larger
    than a single answer sentence, so union inflates the denominator.
    """
    set_a = set(answer_tokens)
    set_b = set(evidence_tokens)
    if not set_a:
        return 0.0
    return len(set_a & set_b) / len(set_a)


def _ngram_overlap(answer_tokens: List[str], evidence_tokens: List[str], n: int = 3) -> float:
    """Compute n-gram overlap ratio between answer and evidence.

    Catches phrase-level matches that token-level Jaccard misses.
    E.g., "led a team of 8" matches "led a team of 8 engineers" via trigrams.
    """
    if len(answer_tokens) < n or len(evidence_tokens) < n:
        return 0.0
    answer_ngrams = set(
        tuple(answer_tokens[i:i + n]) for i in range(len(answer_tokens) - n + 1)
    )
    evidence_ngrams = set(
        tuple(evidence_tokens[i:i + n]) for i in range(len(evidence_tokens) - n + 1)
    )
    if not answer_ngrams:
        return 0.0
    return len(answer_ngrams & evidence_ngrams) / len(answer_ngrams)


def _synonym_expanded_containment(
    answer_tokens: List[str], evidence_tokens_set: frozenset[str]
) -> float:
    """Containment with synonym expansion — catches paraphrased claims.

    E.g., answer says "compensation" and evidence says "salary" → match.
    """
    if not answer_tokens:
        return 0.0
    matched = 0
    for t in answer_tokens:
        if t in evidence_tokens_set:
            matched += 1
            continue
        # Check synonyms
        syns = _SYNONYM_MAP.get(t)
        if syns and syns & evidence_tokens_set:
            matched += 1
    return matched / len(answer_tokens)


def _sentence_critical(sentence: str) -> bool:
    if not sentence:
        return False
    if re.search(r"\d", sentence):
        return True
    if _ENTITY_RE.search(sentence):
        return True
    if _ACRONYM_RE.search(sentence):
        return True
    lowered = sentence.lower()
    # Check single-word modals against split tokens
    lowered_words = lowered.split()
    if any(modal in lowered_words for modal in _MODAL_VERBS if " " not in modal):
        return True
    # Check multi-word modals as substrings
    if any(modal in lowered for modal in _MODAL_VERBS if " " in modal):
        return True
    # Domain-specific critical indicators (medical/lab/financial)
    if any(term in lowered for term in (
        "positive", "negative", "elevated", "decreased", "increased",
        "above normal", "below normal", "high", "low", "abnormal",
    )):
        return True
    # Temporal markers
    if _TEMPORAL_RE.search(sentence):
        return True
    # Comparative markers
    if _COMPARATIVE_RE.search(sentence):
        return True
    # Quantitative markers
    if _QUANTITATIVE_RE.search(sentence):
        return True
    return False


def _key_terms(sentence: str) -> List[str]:
    digits = re.findall(r"\d+(?:\.\d+)?", sentence)
    entities = [m.group(0) for m in _ENTITY_RE.finditer(sentence)]
    acronyms = [m.group(0) for m in _ACRONYM_RE.finditer(sentence)]
    terms = [t.lower() for t in digits + entities + acronyms if t and len(t) >= 2]
    return list(dict.fromkeys(terms))


def _contains_key_terms(sentence_terms: Sequence[str], chunk_text: str) -> bool:
    """Check if any key term (or its synonym or stem-variant) appears in the chunk text."""
    if not sentence_terms or not chunk_text:
        return False
    lowered = chunk_text.lower()
    _chunk_tokens: frozenset[str] | None = None  # lazy-build for stem matching
    for term in sentence_terms:
        t = term.lower()
        # For numeric terms, use word-boundary check to avoid partial matches
        # e.g., "2024" should not match "20241234"
        if t[0].isdigit():
            if re.search(r'\b' + re.escape(t) + r'\b', lowered):
                return True
        else:
            if t in lowered:
                return True
            # Check synonyms for non-numeric terms
            synonyms = _SYNONYM_MAP.get(t)
            if synonyms:
                for syn in synonyms:
                    if syn in lowered:
                        return True
            # Stem-like matching: common prefix ≥5 chars catches plural/tense variants
            if len(t) >= 5:
                _prefix = t[:max(5, len(t) - 3)]
                if _chunk_tokens is None:
                    _chunk_tokens = frozenset(
                        w for w in _TOKEN_RE.findall(lowered) if len(w) >= 5
                    )
                if any(ct[:len(_prefix)] == _prefix for ct in _chunk_tokens):
                    return True
    return False


def _check_negation_conflict(sentence: str, chunk_texts_lower: Sequence[str]) -> bool:
    """Return True if there is a negation mismatch between sentence and evidence.

    E.g., answer says 'positive' but evidence says 'NOT positive' or 'negative'.
    Also catches 'has X' in answer when evidence says 'does not have X'.
    """
    sent_lower = sentence.lower()
    sent_has_negation = bool(_NEGATION_PREFIX_RE.search(sent_lower))

    # Extract content words near negation to detect conflict
    for chunk_lower in chunk_texts_lower:
        # Case 1: sentence is affirmative, evidence has negation of same claim
        if not sent_has_negation:
            # Check if evidence negates something the sentence affirms
            for match in _NEGATION_PREFIX_RE.finditer(chunk_lower):
                # Get the ~40 chars after the negation
                start = match.end()
                window = chunk_lower[start:start + 40].strip().split()
                if not window:
                    continue
                # If negated words in evidence overlap with sentence content tokens
                negated_terms = set(w.strip(".,;:") for w in window[:4] if len(w) > 2)
                sent_tokens = set(sent_lower.split())
                if negated_terms & sent_tokens:
                    return True
        else:
            # Case 2: sentence has negation, but evidence affirms (no negation nearby)
            # Extract the content word after our negation
            for match in _NEGATION_PREFIX_RE.finditer(sent_lower):
                start = match.end()
                window = sent_lower[start:start + 40].strip().split()
                if not window:
                    continue
                negated_in_sent = set(w.strip(".,;:") for w in window[:4] if len(w) > 2)
                # Check if evidence has these words without negation
                for word in negated_in_sent:
                    if word in chunk_lower:
                        # Verify evidence does NOT also negate this word
                        # Find word in chunk and check preceding context
                        idx = chunk_lower.find(word)
                        if idx >= 0:
                            preceding = chunk_lower[max(0, idx - 30):idx]
                            if not _NEGATION_PREFIX_RE.search(preceding):
                                return True
    return False


def _adaptive_threshold(base: float, answer_len: int) -> float:
    """Adjust grounding threshold based on answer length."""
    if answer_len < 100:
        return base + 0.08  # short claims need higher overlap
    elif answer_len > 500:
        return base - 0.03  # natural dilution in long text
    return base


def _validate_table_line(
    sentence: str,
    all_chunks_lower: str,
    chunk_texts_lower: Sequence[str],
) -> float:
    """Validate a table line's content against evidence.

    Returns a score: 1.0 = fully supported, 0.5 = partially, 0.0 = unsupported.
    """
    stripped = sentence.strip()

    # Separator rows are always fine
    if _TABLE_SEP_RE.match(stripped):
        return 1.0

    # Must have >=2 pipe-separated cells to auto-pass
    cells = [c.strip() for c in stripped.strip("|").split("|") if c.strip()]
    if len(cells) < 2:
        return 0.5  # single-cell or malformed — don't auto-pass

    # Check numbers and names in cells against evidence
    unsupported_count = 0
    checkable_count = 0
    for cell in cells:
        cell_stripped = cell.strip()
        if not cell_stripped or _TABLE_SEP_RE.match("|" + cell_stripped + "|"):
            continue
        # Extract numbers from cell
        numbers = re.findall(r"\d+(?:[.,]\d+)*", cell_stripped)
        # Extract capitalized names
        names = _ENTITY_RE.findall(cell_stripped)

        items_to_check = numbers + [n.lower() for n in names]
        if not items_to_check:
            continue

        checkable_count += 1
        found_any = False
        for item in items_to_check:
            t = item.lower().replace(",", "")
            if t[0:1].isdigit():
                if re.search(r'\b' + re.escape(t) + r'\b', all_chunks_lower):
                    found_any = True
                    break
            else:
                if t in all_chunks_lower:
                    found_any = True
                    break
                # Check synonyms
                synonyms = _SYNONYM_MAP.get(t)
                if synonyms and any(s in all_chunks_lower for s in synonyms):
                    found_any = True
                    break
                # Stem-like matching for table cell names
                if len(t) >= 5:
                    _prefix = t[:max(5, len(t) - 3)]
                    if _prefix in all_chunks_lower:
                        found_any = True
                        break

        if not found_any:
            unsupported_count += 1

    if checkable_count == 0:
        return 1.0  # header row with no verifiable content
    if unsupported_count == 0:
        return 1.0
    # Require ≥75% cell support for tables — half-correct rows look credible
    # but contain fabrication. Stricter than prose grounding.
    support_ratio = 1.0 - (unsupported_count / checkable_count)
    if support_ratio >= 0.75:
        return 0.8
    if support_ratio >= 0.5:
        # Proportional scoring for marginal support (0.5→0.3, 0.75→0.5)
        return 0.3 + (support_ratio - 0.5) * 0.8
    if support_ratio >= 0.25:
        return 0.15  # low but not zero — some data is grounded
    return 0.0


_TRANSITION_RE = re.compile(
    r"^\s*(?:"
    r"in\s+(?:summary|conclusion|addition|particular|contrast)"
    r"|to\s+summarize"
    r"|overall"
    r"|furthermore|moreover|however|therefore|consequently"
    r"|the\s+(?:following|above|below)\s+(?:table|list|information|details)"
    r"|here\s+(?:is|are)\s+(?:the|a)\s+"
    r"|as\s+(?:shown|noted|mentioned|discussed)\s+(?:above|below|earlier)"
    r"|key\s+(?:findings|takeaways|points|highlights)"
    r"|additional\s+(?:details|information|notes)"
    r")\b",
    re.IGNORECASE,
)


def _is_transition_sentence(sentence: str) -> bool:
    """Check if sentence is a PURE transition/connector with no factual claims.

    Only skip grounding for sentences where the transition IS the entire content.
    If there's a factual claim after the transition marker, ground it.
    """
    stripped = sentence.strip()
    # Markdown headers that are just section labels
    if re.match(r"^#{1,4}\s+\w+", stripped) and len(stripped) < 60:
        return True
    # Check for transition prefix
    if not _TRANSITION_RE.match(stripped):
        return False
    # Strip the transition marker and check what remains
    remainder = _TRANSITION_RE.sub("", stripped).strip().strip(",:;—")
    # If nothing remains after stripping, it's a pure transition
    if not remainder or len(remainder) < 10:
        return True
    # If remainder has numbers or entities, it's a factual claim — don't skip
    if re.search(r"\d", remainder) or _ENTITY_RE.search(remainder):
        return False
    # Short remainder without factual content — still a transition
    if len(remainder) < 25:
        return True
    return False


_DOMAIN_THRESHOLDS = {
    "medical": (0.35, 0.50),  # (base, critical) — strict for medical accuracy
    "legal": (0.28, 0.40),    # tightened — catch unsupported legal claims
    "invoice": (0.30, 0.42),  # structured: numbers must match evidence
    "hr": (0.28, 0.40),       # moderate strictness
    "policy": (0.28, 0.40),   # policy claims need evidence support
}


def evaluate_grounding(
    answer: str,
    chunk_texts: Sequence[str],
    *,
    support_threshold: float = 0.30,
    domain: str = "",
) -> GroundingResult:
    """Evaluate how well the answer is grounded in evidence chunks.

    Uses tiered thresholds: critical sentences (containing numbers or named
    entities) require higher token overlap than regular sentences.
    Domain-adaptive thresholds when domain is specified.
    Negation-aware: detects conflicts between answer claims and evidence.
    Threshold adapts to answer length (stricter for short, lenient for long).
    """
    # Apply domain-specific thresholds if available
    if domain and domain.lower() in _DOMAIN_THRESHOLDS:
        base, crit = _DOMAIN_THRESHOLDS[domain.lower()]
        support_threshold = base

    # Enhancement 3: adaptive threshold by answer length
    support_threshold = _adaptive_threshold(support_threshold, len(answer))

    # Detect structured content (tables/lists) — these need block-level grounding,
    # not sentence-level Jaccard which catastrophically fails on short cell rows
    _answer_has_table = _has_markdown_table(answer)

    sentences = _split_sentences(answer)
    if not sentences:
        return GroundingResult(0.0, 0.0, [], [], [])

    chunk_tokens = [_tokenize(text) for text in chunk_texts]
    chunk_lowers = [text.lower() for text in chunk_texts]
    # Pre-build combined chunk text for fast key-term lookup
    _all_chunks_lower = " ".join(chunk_lowers)

    supported_sentences: List[str] = []
    unsupported_sentences: List[str] = []
    scores: List[float] = []

    critical_total = 0
    critical_supported = 0

    # Tiered threshold: critical sentences need stronger overlap
    # Use domain-specific critical threshold if available
    if domain and domain.lower() in _DOMAIN_THRESHOLDS:
        _critical_threshold = _DOMAIN_THRESHOLDS[domain.lower()][1]
    else:
        _critical_threshold = max(support_threshold, 0.35)

    prev_score = 0.0  # Track previous sentence score for coherence bonus

    for sentence in sentences:
        tokens = _tokenize(sentence)
        if not tokens:
            scores.append(1.0)
            supported_sentences.append(sentence)
            prev_score = 1.0
            continue

        # Enhancement 4: table line validation (replaces blind auto-pass)
        if _answer_has_table and _is_table_or_list_line(sentence):
            table_score = _validate_table_line(sentence, _all_chunks_lower, chunk_lowers)
            scores.append(table_score)
            if table_score >= 0.5:
                supported_sentences.append(sentence)
            else:
                unsupported_sentences.append(sentence)
            continue

        # Skip transition/connector sentences — these don't carry factual claims
        # and shouldn't penalize grounding score
        if _is_transition_sentence(sentence):
            scores.append(0.8)  # Neutral score — not penalized, not rewarded
            supported_sentences.append(sentence)
            continue

        key_terms = _key_terms(sentence)
        is_critical = _sentence_critical(sentence)
        # Use stricter threshold for sentences with numbers/entities
        effective_threshold = _critical_threshold if is_critical else support_threshold

        # Multi-signal grounding: combine Jaccard, containment, n-gram, synonym
        best_jaccard = 0.0
        best_containment = 0.0
        best_ngram = 0.0
        for idx, chunk in enumerate(chunk_tokens):
            j = _jaccard(tokens, chunk)
            c = _containment(tokens, chunk)
            ng = _ngram_overlap(tokens, chunk, n=3) if len(tokens) >= 3 else 0.0
            if j > best_jaccard:
                best_jaccard = j
            if c > best_containment:
                best_containment = c
            if ng > best_ngram:
                best_ngram = ng

        # Synonym-aware containment against combined evidence
        _all_evidence_tokens = frozenset(
            t for cl in chunk_tokens for t in cl
        )
        syn_containment = _synonym_expanded_containment(tokens, _all_evidence_tokens)

        # Composite score: best of multiple signals
        # Containment is most reliable for grounding (answer terms in evidence)
        # N-gram catches phrase-level matches; synonym catches paraphrases
        best = max(
            best_jaccard,
            best_containment * 0.9,   # slight discount vs exact Jaccard
            best_ngram * 0.85,        # phrase match is strong signal
            syn_containment * 0.8,    # synonym match is softer signal
        )

        # Enhancement 2: negation conflict check — apply 0.5x penalty
        if _check_negation_conflict(sentence, chunk_lowers):
            best *= 0.5

        # Paragraph coherence bonus: if the previous sentence was well-grounded,
        # the current sentence likely continues the same grounded thought.
        # This reduces false hallucination flags on continuation sentences like
        # "He also led the cloud migration project." following a well-grounded claim.
        if prev_score >= 0.6 and best < effective_threshold:
            coherence_bonus = min(0.15, (prev_score - 0.5) * 0.3)
            best = best + coherence_bonus

        # Fast path: check key terms against combined text first
        key_match = _contains_key_terms(key_terms, _all_chunks_lower) if key_terms else False
        supported = best >= effective_threshold or key_match

        # Even if key_match passed, negation conflict should override
        if key_match and _check_negation_conflict(sentence, chunk_lowers):
            supported = False

        scores.append(best)
        prev_score = best
        if supported:
            supported_sentences.append(sentence)
        else:
            unsupported_sentences.append(sentence)
        if is_critical:
            critical_total += 1
            if supported:
                critical_supported += 1

    supported_ratio = len(supported_sentences) / max(len(sentences), 1)
    if critical_total:
        critical_ratio = critical_supported / max(critical_total, 1)
    else:
        critical_ratio = 1.0

    return GroundingResult(
        supported_ratio=round(supported_ratio, 4),
        critical_supported_ratio=round(critical_ratio, 4),
        supported_sentences=supported_sentences,
        unsupported_sentences=unsupported_sentences,
        sentence_scores=scores,
    )
