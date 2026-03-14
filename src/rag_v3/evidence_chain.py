"""Chain-of-evidence reasoning for complex queries.

Analyzes retrieved chunks to identify supporting facts, contradictions,
and information gaps BEFORE the LLM generation call. The structured
evidence chain is injected into the prompt for more reasoned answers.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_QUERY_KEYWORDS_RE = re.compile(r"[A-Za-z]{3,}")
_NUMERIC_KV_RE = re.compile(
    r"([A-Za-z][A-Za-z\s]{0,40}?):\s*[\$£€₹¥]?\s*([\d,]+(?:\.\d+)?)\s*"
    r"(%|years?|months?|days?|hours?|weeks?"
    r"|kg|lbs?|mg|ml|oz|grams?"
    r"|usd|eur|gbp|inr|[kKmMbB])?",
    re.IGNORECASE,
)


@dataclass
class EvidenceFact:
    text: str
    source: str
    chunk_id: str
    relevance: float


@dataclass
class TopicGroup:
    """Group of facts sharing a dominant topic."""
    topic: str
    facts: List[EvidenceFact] = field(default_factory=list)


@dataclass
class Contradiction:
    """Conflicting values for the same label across documents."""
    label: str
    values: List[Tuple[str, str]]  # (value, source)


@dataclass
class NumericStat:
    """Aggregate statistics for a numeric label."""
    label: str
    values: List[float]
    sources: List[str]

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def total(self) -> float:
        return sum(self.values)

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)

    @property
    def minimum(self) -> float:
        return min(self.values) if self.values else 0.0

    @property
    def maximum(self) -> float:
        return max(self.values) if self.values else 0.0


@dataclass
class EvidenceChain:
    query: str
    supporting_facts: List[EvidenceFact] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    num_documents: int = 0
    topic_groups: List[TopicGroup] = field(default_factory=list)
    contradictions: List[Contradiction] = field(default_factory=list)
    numeric_stats: List[NumericStat] = field(default_factory=list)

    def render_for_prompt(self) -> str:
        """Render the evidence chain as structured context for the LLM prompt."""
        parts: List[str] = []

        # KEY FINDINGS summary — gives LLM a quick orientation before detailed evidence
        if self.supporting_facts:
            top_facts = self.supporting_facts[:3]
            key_lines = []
            for fact in top_facts:
                # Extract first sentence as the key finding
                _first_sent = fact.text.split(". ")[0].strip()
                if len(_first_sent) > 120:
                    _first_sent = _first_sent[:117] + "..."
                key_lines.append(f"  • {_first_sent}")
            if key_lines:
                parts.append("KEY FINDINGS:")
                parts.extend(key_lines)

        if self.supporting_facts:
            if self.topic_groups:
                parts.append("\nEVIDENCE FOUND (grouped by topic):")
                for group in self.topic_groups:
                    parts.append(f"\n  [{group.topic.title()}]")
                    for fact in group.facts:
                        parts.append(f"    - {fact.text} (Source: {fact.source})")
            else:
                parts.append("\nEVIDENCE FOUND:")
                for i, fact in enumerate(self.supporting_facts, 1):
                    parts.append(f"  [{i}] {fact.text} (Source: {fact.source})")

        if self.numeric_stats:
            parts.append("\nSTATISTICS FROM EVIDENCE:")
            for stat in self.numeric_stats:
                if stat.count >= 2:
                    parts.append(
                        f"  - {stat.label}: count={stat.count}, "
                        f"sum={stat.total:.2f}, avg={stat.average:.2f}, "
                        f"min={stat.minimum:.2f}, max={stat.maximum:.2f}"
                    )
                else:
                    parts.append(f"  - {stat.label}: {stat.values[0]:.2f}")

        if self.contradictions:
            parts.append("\nCONTRADICTIONS ACROSS DOCUMENTS:")
            for c in self.contradictions:
                vals = ", ".join(f"{v} ({s})" for v, s in c.values)
                parts.append(f"  - {c.label}: {vals}")

        if self.gaps:
            parts.append("\nINFORMATION NOT FOUND IN DOCUMENTS:")
            for gap in self.gaps:
                parts.append(f"  - {gap}")

        if self.num_documents > 1:
            # Show per-document fact distribution for multi-doc awareness
            doc_fact_counts: dict[str, int] = {}
            for fact in self.supporting_facts:
                doc_fact_counts[fact.source] = doc_fact_counts.get(fact.source, 0) + 1
            if doc_fact_counts:
                dist = ", ".join(f"{doc}: {cnt}" for doc, cnt in sorted(doc_fact_counts.items(), key=lambda x: -x[1]))
                parts.append(f"\nDocuments analyzed: {self.num_documents} (facts: {dist})")
            else:
                parts.append(f"\nDocuments analyzed: {self.num_documents}")

        # Evidence quality summary — helps LLM calibrate confidence
        if self.supporting_facts:
            avg_relevance = sum(f.relevance for f in self.supporting_facts) / len(self.supporting_facts)
            quality = "strong" if avg_relevance >= 0.5 else "moderate" if avg_relevance >= 0.25 else "weak"
            has_gaps = len(self.gaps) > 0
            has_conflicts = len(self.contradictions) > 0
            quality_notes = []
            if has_gaps:
                quality_notes.append(f"{len(self.gaps)} gap(s)")
            if has_conflicts:
                quality_notes.append(f"{len(self.contradictions)} conflict(s)")
            quality_suffix = f" ({', '.join(quality_notes)})" if quality_notes else ""
            parts.append(f"\nEvidence quality: {quality} (avg relevance: {avg_relevance:.2f}){quality_suffix}")

        return "\n".join(parts)


def _group_facts_by_topic(facts: List[EvidenceFact]) -> List[TopicGroup]:
    """Cluster facts by dominant nouns shared between them."""
    if len(facts) < 3:
        return []

    # Extract dominant nouns per fact (non-stopword words >= 4 chars)
    fact_nouns: List[set] = []
    for fact in facts:
        words = set(
            w.lower() for w in _QUERY_KEYWORDS_RE.findall(fact.text)
            if len(w) >= 4 and w.lower() not in _STOPWORDS
        )
        fact_nouns.append(words)

    # Find frequently-occurring nouns as topic seeds
    all_nouns: Counter = Counter()
    for nouns in fact_nouns:
        all_nouns.update(nouns)

    # Topics: nouns that appear in >=2 facts
    topic_seeds = [noun for noun, count in all_nouns.most_common(10) if count >= 2]
    if not topic_seeds:
        return []

    groups: Dict[str, TopicGroup] = {}
    assigned: set = set()

    for topic in topic_seeds[:5]:
        group = TopicGroup(topic=topic)
        for i, (fact, nouns) in enumerate(zip(facts, fact_nouns)):
            if i not in assigned and topic in nouns:
                group.facts.append(fact)
                assigned.add(i)
        if group.facts:
            groups[topic] = group

    # Add unassigned facts to an "other" group
    other_facts = [f for i, f in enumerate(facts) if i not in assigned]
    if other_facts:
        groups["other"] = TopicGroup(topic="other", facts=other_facts)

    return list(groups.values())


def _detect_contradictions(facts: List[EvidenceFact]) -> List[Contradiction]:
    """Find conflicting values for the same label across different documents."""
    if len(facts) < 2:
        return []

    label_values: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for fact in facts:
        # Use search() not match() — fact.text is raw chunk content,
        # "Label: Value" patterns can appear anywhere, not just at the start
        match = _NUMERIC_KV_RE.search(fact.text)
        if match:
            raw_label = match.group(1).strip().lower()
            # Normalize label: strip common prefix words to compare field names
            # e.g., "the document states salary" → "salary"
            _PREFIX_WORDS = {"the", "a", "an", "this", "that", "its", "our", "their",
                             "according", "to", "per", "from", "in", "of", "on", "by",
                             "document", "documents", "states", "records", "report",
                             "shows", "indicates", "lists", "mentions", "notes"}
            label_words = [w for w in raw_label.split() if w not in _PREFIX_WORDS]
            label = " ".join(label_words) if label_words else raw_label
            value = match.group(2).strip()
            label_values[label].append((value, fact.source))

    contradictions: List[Contradiction] = []
    for label, entries in label_values.items():
        # Only flag as contradiction if values differ AND come from different sources
        unique_vals = set(v for v, _ in entries)
        unique_sources = set(s for _, s in entries)
        if len(unique_vals) > 1 and len(unique_sources) > 1:
            contradictions.append(Contradiction(label=label, values=entries))

    return contradictions[:5]


def _compute_numeric_stats(facts: List[EvidenceFact]) -> List[NumericStat]:
    """Extract Label: $Number patterns and compute aggregate statistics."""
    label_nums: Dict[str, NumericStat] = {}

    for fact in facts:
        match = _NUMERIC_KV_RE.search(fact.text)
        if match:
            label = match.group(1).strip()
            raw_val = match.group(2).replace(",", "")
            try:
                num = float(raw_val)
            except ValueError:
                continue
            key = label.lower()
            if key not in label_nums:
                label_nums[key] = NumericStat(label=label, values=[], sources=[])
            label_nums[key].values.append(num)
            label_nums[key].sources.append(fact.source)

    return [stat for stat in label_nums.values() if stat.count >= 1][:10]


def build_evidence_chain(query: str, chunks: List[Any]) -> EvidenceChain:
    """Analyze chunks to build a structured evidence chain for the query."""
    query_keywords = set(
        w.lower() for w in _QUERY_KEYWORDS_RE.findall(query)
        if len(w) > 3 and w.lower() not in _STOPWORDS
    )

    supporting: List[EvidenceFact] = []
    doc_names: set[str] = set()

    # Extract named entities from query for entity-relevance boost
    _entity_re = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b")
    query_entities = {m.group().lower() for m in _entity_re.finditer(query) if len(m.group()) > 2}

    for chunk in chunks:
        text = (getattr(chunk, "text", "") or "").strip()
        if not text:
            continue

        meta = getattr(chunk, "meta", None) or {}
        source = meta.get("source_name") or getattr(
            getattr(chunk, "source", None), "document_name", "Unknown"
        )
        doc_names.add(source)

        chunk_words = set(w.lower() for w in _QUERY_KEYWORDS_RE.findall(text))
        overlap = query_keywords & chunk_words
        relevance = len(overlap) / max(len(query_keywords), 1)

        # Boost relevance for chunks mentioning query entities
        if query_entities:
            text_lower = text.lower()
            entity_hits = sum(1 for e in query_entities if e in text_lower)
            relevance = min(1.0, relevance + entity_hits * 0.15)

        # Incorporate rerank/vector score as a floor — chunks that scored
        # high in retrieval are relevant even if keyword overlap is low
        _chunk_score = getattr(chunk, "score", 0) or 0
        if _chunk_score > 0.5:
            relevance = max(relevance, _chunk_score * 0.6)

        if relevance > 0.1 or _chunk_score > 0.5:
            # Sentence-boundary truncation instead of hard cut
            if len(text) > 300:
                _trunc = text[:300]
                _last_period = max(_trunc.rfind(". "), _trunc.rfind(".\n"))
                fact_text = text[:_last_period + 1] if _last_period > 150 else _trunc
            else:
                fact_text = text
            supporting.append(EvidenceFact(
                text=fact_text,
                source=source,
                chunk_id=getattr(chunk, "id", ""),
                relevance=relevance,
            ))

    # Identify gaps: query keywords not covered by any supporting fact
    covered_keywords: set[str] = set()
    for fact in supporting:
        fact_words = set(w.lower() for w in _QUERY_KEYWORDS_RE.findall(fact.text))
        covered_keywords |= fact_words

    uncovered = query_keywords - covered_keywords - _STOPWORDS
    # Prioritize entity gaps: match uncovered keywords against original query
    # entities (which preserve capitalization), then generic terms
    entity_gap_words = {e.lower() for e in query_entities} & uncovered
    term_gap_words = uncovered - entity_gap_words
    gaps = []
    for kw in sorted(entity_gap_words):
        # Find original-case version from query_entities
        original = next((e for e in query_entities if e.lower() == kw), kw)
        gaps.append(f"No evidence found about: {original}")
    for kw in sorted(term_gap_words):
        if len(kw) > 3:
            gaps.append(f"No evidence found about: {kw}")

    # Sort supporting facts by relevance
    supporting.sort(key=lambda f: f.relevance, reverse=True)

    # Deduplicate near-identical facts (same content from different chunk boundaries)
    supporting = _deduplicate_facts(supporting)
    supporting = supporting[:12]  # Allow slightly more after dedup

    # Enhanced analysis: topic grouping, contradictions, statistics
    topic_groups = _group_facts_by_topic(supporting)
    contradictions = _detect_contradictions(supporting)
    numeric_stats = _compute_numeric_stats(supporting)

    return EvidenceChain(
        query=query,
        supporting_facts=supporting,
        gaps=gaps[:5],
        num_documents=len(doc_names),
        topic_groups=topic_groups,
        contradictions=contradictions,
        numeric_stats=numeric_stats,
    )


def _deduplicate_facts(facts: List[EvidenceFact], threshold: float = 0.75) -> List[EvidenceFact]:
    """Remove near-duplicate facts based on word overlap.

    Keeps the first (highest-relevance) version of each fact cluster.
    """
    if len(facts) <= 1:
        return facts

    kept: List[EvidenceFact] = []
    kept_words: List[set] = []

    for fact in facts:
        words = set(w.lower() for w in _QUERY_KEYWORDS_RE.findall(fact.text) if len(w) > 2)
        if not words:
            kept.append(fact)
            kept_words.append(words)
            continue

        is_dup = False
        for existing_words in kept_words:
            if not existing_words:
                continue
            intersection = len(words & existing_words)
            union = len(words | existing_words)
            if union > 0 and intersection / union >= threshold:
                is_dup = True
                break

        if not is_dup:
            kept.append(fact)
            kept_words.append(words)

    return kept


_STOPWORDS = frozenset({
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "has", "have", "been", "this", "that",
    "with", "they", "from", "does", "about", "tell", "show", "give",
    "many", "much", "some", "than", "them", "then", "these", "those",
    "each", "other", "into", "through", "during", "before", "after",
    "above", "below", "between", "same", "different",
})
