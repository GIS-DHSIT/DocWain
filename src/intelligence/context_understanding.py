"""ML-based context understanding for document intelligence.

Uses embeddings and semantic analysis to deeply understand document content
before LLM extraction. This module is document-type agnostic and works with
any domain (HR, medical, legal, invoice, policy, generic).

Key capabilities:
  - Semantic chunk clustering: groups chunks by topic/section using cosine similarity
  - Query-evidence alignment: scores how well each chunk addresses the query
  - Entity salience: detects which entities are most relevant to the query
  - Cross-document relationships: finds connections between documents
  - Context distillation: produces a structured context summary for LLM prompts

The output is a ContextUnderstanding object that enriches the LLM prompt with
structured knowledge, enabling more accurate and intelligent responses.
"""
from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = get_logger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────
_CLUSTER_SIMILARITY_THRESHOLD = 0.72  # cosine sim to merge into same cluster
_MAX_CLUSTERS = 8
_MIN_ALIGNMENT_SCORE = 0.25  # min query-chunk alignment to consider relevant
_MAX_CONTEXT_ENTITIES = 12
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b")
_KV_RE = re.compile(r"^(.{3,40})\s*[:]\s+(.{2,})", re.MULTILINE)
_NUMBER_RE = re.compile(r"\b\d[\d,.]+\b")
_DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\w+ \d{1,2},?\s*\d{4}|"
    r"\d{4}[/\-]\d{1,2}[/\-]\d{1,2})\b"
)

# Multi-hop query patterns — queries needing reasoning across multiple chunks
_MULTI_HOP_PATTERNS = [
    re.compile(r"\b(?:compare|contrast|difference|similarities)\b.*\b(?:and|vs|versus|with)\b", re.I),
    re.compile(r"\b(?:relationship|connection|correlation)\s+between\b", re.I),
    re.compile(r"\b(?:how\s+does|how\s+do)\b.*\b(?:affect|impact|influence|relate)\b", re.I),
    re.compile(r"\b(?:based\s+on|considering|given\s+that)\b.*\b(?:what|how|should|would)\b", re.I),
    re.compile(r"\b(?:if|when|assuming)\b.*\b(?:then|would|should|will)\b", re.I),
    # Aggregation queries (total, average, sum across docs)
    re.compile(r"\b(?:total|sum|average|aggregate|combined|overall)\b.*\b(?:across|all|every|each)\b", re.I),
    # Conditional reasoning
    re.compile(r"\b(?:which|who)\b.*\b(?:best|most|least|highest|lowest)\b.*\b(?:for|given|considering)\b", re.I),
    # Synthesis queries requiring multiple evidence pieces
    re.compile(r"\b(?:summarize|synthesize|overview)\b.*\b(?:all|both|across|multiple)\b", re.I),
    # Cross-document patterns
    re.compile(r"\b(?:across|between|among)\s+(?:all\s+)?(?:documents?|files?|sources?|records?)\b", re.I),
    # Implicit multi-hop: "X and Y" entity comparison
    re.compile(r"\b[A-Z][a-z]+\s+(?:and|vs\.?|versus)\s+[A-Z][a-z]+\b"),
    # Qualification/eligibility queries requiring cross-referencing
    re.compile(r"\b(?:eligible|qualify|meets?\s+(?:the\s+)?(?:criteria|requirements?))\b", re.I),
]

# Entity alias patterns for cross-document resolution
_TITLE_PREFIXES = re.compile(
    r"^(?:Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Prof\.?|Sir|Dame|Rev\.?|Hon\.?)\s+", re.I
)
_SUFFIX_RE = re.compile(
    r",?\s+(?:Jr\.?|Sr\.?|III|IV|II|Ph\.?D\.?|M\.?D\.?|Esq\.?|CPA|MBA)$", re.I
)

# Common stopwords to exclude from topic extraction
_STOP = frozenset({
    "the", "and", "for", "are", "was", "were", "has", "have", "had",
    "this", "that", "with", "from", "not", "but", "all", "can", "will",
    "been", "being", "more", "than", "each", "which", "their", "also",
    "into", "some", "only", "other", "very", "just", "about", "would",
    "could", "should", "these", "those", "between", "through", "during",
})

# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class TopicCluster:
    """A semantic cluster of related chunks."""
    topic_label: str
    chunk_indices: List[int]
    representative_terms: List[str]
    avg_score: float = 0.0

@dataclass
class EntitySalience:
    """An entity with its relevance score to the query."""
    name: str
    entity_type: str  # "person", "org", "number", "date", "term"
    salience: float  # 0.0 - 1.0
    mentions: int
    source_documents: List[str]

@dataclass
class QueryAlignment:
    """How well a chunk aligns with the query."""
    chunk_index: int
    alignment_score: float  # 0.0 - 1.0
    matching_aspects: List[str]

@dataclass
class DocumentRelationship:
    """Relationship between two documents."""
    doc_a: str
    doc_b: str
    relationship_type: str  # "similar_topic", "shared_entities", "complementary"
    strength: float

@dataclass
class StructuredFact:
    """A key-value fact extracted from document content."""
    key: str
    value: str
    source_doc: str
    confidence: float

@dataclass
class TemporalSpan:
    """A temporal reference extracted from document content."""
    text: str  # raw date/period text
    context: str  # surrounding sentence
    source_doc: str
    sort_key: int = 0  # numeric key for ordering (e.g., YYYYMMDD)

@dataclass
class CausalLink:
    """A cause-effect relationship detected in content."""
    cause: str
    effect: str
    source_doc: str
    marker: str  # the causal marker word that triggered detection

@dataclass
class ContextUnderstanding:
    """Complete context understanding for a query + chunks."""
    topic_clusters: List[TopicCluster]
    entity_salience: List[EntitySalience]
    query_alignments: List[QueryAlignment]
    document_relationships: List[DocumentRelationship]
    structured_facts: List[StructuredFact]
    content_summary: str
    document_count: int
    total_chunks: int
    dominant_domain: str
    key_topics: List[str]
    # Temporal reasoning
    temporal_spans: List[TemporalSpan] = field(default_factory=list)
    causal_links: List[CausalLink] = field(default_factory=list)
    # Confidence metrics for context quality assessment
    alignment_quality: str = "unknown"  # "strong", "moderate", "weak"
    context_confidence: float = 0.0  # 0.0-1.0 overall confidence
    # Multi-hop reasoning flag
    is_multi_hop: bool = False
    # Entity alias map: canonical_name → set of aliases found across docs
    entity_aliases: Dict[str, List[str]] = field(default_factory=dict)
    # Cross-document fact conflicts: [(key, value_a, doc_a, value_b_info)]
    fact_conflicts: List[Tuple[str, str, str, str]] = field(default_factory=list)

    def to_prompt_section(self, intent: str = "") -> str:
        """Render as a structured section for LLM prompt injection.

        Intent-aware filtering: only include sections relevant to the query
        intent, reducing noise that dilutes LLM focus.
        """
        parts: List[str] = []
        _intent = (intent or "").lower()

        # Define which sections are useful for each intent category
        _SIMPLE_INTENTS = {"factual", "contact", "extract", "qa", "list", "detail"}
        _ANALYTICAL_INTENTS = {"comparison", "ranking", "analytics", "cross_document", "reasoning", "analyze"}
        _TEMPORAL_INTENTS = {"timeline", "history"}
        _is_simple = _intent in _SIMPLE_INTENTS
        _is_analytical = _intent in _ANALYTICAL_INTENTS
        _is_temporal = _intent in _TEMPORAL_INTENTS

        if self.content_summary:
            confidence_note = ""
            if self.alignment_quality == "weak":
                confidence_note = " Note: Evidence alignment with the query is limited — answer with caution."
            parts.append(f"DOCUMENT INTELLIGENCE:\n{self.content_summary}{confidence_note}")

        # Key topics: useful for analytical/summary, skip for simple factual
        if self.key_topics and not _is_simple:
            parts.append(f"KEY TOPICS: {', '.join(self.key_topics[:6])}")

        # Entities: always useful — helps LLM identify who/what
        if self.entity_salience:
            salient = [
                f"{e.name} ({e.entity_type}, relevance: {e.salience:.0%})"
                for e in self.entity_salience[:8]
                if e.salience >= 0.3
            ]
            if salient:
                parts.append(f"KEY ENTITIES: {'; '.join(salient)}")

        # Facts: always useful — direct answers to queries
        if self.structured_facts:
            top_facts = sorted(
                self.structured_facts, key=lambda f: f.confidence, reverse=True
            )[:10]
            fact_lines = [f"  - {f.key}: {f.value}" for f in top_facts]
            if fact_lines:
                parts.append("EXTRACTED FACTS:\n" + "\n".join(fact_lines))

        # Document relationships: only for multi-doc analytical queries
        if self.document_relationships and (_is_analytical or self.document_count > 1):
            rels = [
                f"  - {r.doc_a} ↔ {r.doc_b}: {r.relationship_type}"
                for r in self.document_relationships[:5]
            ]
            if rels:
                parts.append("DOCUMENT RELATIONSHIPS:\n" + "\n".join(rels))

        # Topic clusters: only for analytical/summary intents
        if self.topic_clusters and len(self.topic_clusters) > 1 and not _is_simple:
            cluster_desc = [
                f"  - {c.topic_label}: {', '.join(c.representative_terms[:4])}"
                for c in self.topic_clusters[:5]
            ]
            parts.append("CONTENT STRUCTURE:\n" + "\n".join(cluster_desc))

        # Timeline: only for temporal/analytical intents
        if self.temporal_spans and (_is_temporal or _is_analytical):
            sorted_spans = sorted(self.temporal_spans, key=lambda s: s.sort_key)
            timeline_lines = [
                f"  - {s.text}: {s.context[:120]}" for s in sorted_spans[:8]
            ]
            parts.append("TIMELINE FROM DOCUMENTS:\n" + "\n".join(timeline_lines))

        # Causal links: only for reasoning/analytical intents
        if self.causal_links and (_is_analytical or _intent == "reasoning"):
            causal_lines = [
                f"  - {cl.cause} → {cl.effect}" for cl in self.causal_links[:5]
            ]
            parts.append("CAUSE-EFFECT RELATIONSHIPS:\n" + "\n".join(causal_lines))

        # Entity aliases for cross-document entity resolution
        if self.entity_aliases:
            alias_lines = []
            for canonical, aliases in list(self.entity_aliases.items())[:6]:
                if aliases:
                    alias_lines.append(f"  - {canonical} (also: {', '.join(aliases[:3])})")
            if alias_lines:
                parts.append("ENTITY ALIASES (same person/org across documents):\n" + "\n".join(alias_lines))

        # Numeric fact prominence for analytical intents
        if self.structured_facts and (_is_analytical or _intent in {"factual", "detail", "extract"}):
            numeric_facts = [
                f for f in self.structured_facts
                if f.value and re.search(r'\d', f.value)
            ]
            _min_numeric = 1 if _intent in {"factual", "detail", "extract"} else 2
            if numeric_facts and len(numeric_facts) >= _min_numeric:
                num_lines = [
                    f"  - {f.key}: {f.value} ({f.source_doc})"
                    for f in sorted(numeric_facts, key=lambda x: x.confidence, reverse=True)[:6]
                ]
                parts.append("KEY NUMERIC FACTS (use exact values from evidence):\n" + "\n".join(num_lines))

        # Multi-hop reasoning guidance
        if self.is_multi_hop:
            parts.append(
                "REASONING GUIDANCE: This query requires multi-hop reasoning — "
                "synthesize information across multiple evidence segments. "
                "Do not answer from a single segment alone; connect facts from different parts."
            )

        # Multi-document completeness guidance
        if self.document_count > 1 and (_is_analytical or _intent in {"comparison", "ranking", "cross_document"}):
            parts.append(
                f"COMPLETENESS: Cover ALL {self.document_count} documents in your response. "
                "Do not skip any source — each document may contain unique information."
            )

        # Query decomposition hint for complex multi-aspect queries
        if self.is_multi_hop and self.key_topics and len(self.key_topics) >= 2:
            parts.append(
                f"QUERY DECOMPOSITION: This query involves multiple aspects "
                f"({', '.join(self.key_topics[:4])}). Address each aspect separately "
                "before synthesizing your final answer."
            )

        # Cross-document fact conflicts — alert LLM to present both values
        if self.fact_conflicts:
            conflict_lines = ["CONFLICTING FACTS ACROSS DOCUMENTS (present BOTH values):"]
            for key, val_a, doc_a, val_b_info in self.fact_conflicts[:5]:
                conflict_lines.append(f"  - {key}: '{val_a}' ({doc_a}) vs {val_b_info}")
            parts.append("\n".join(conflict_lines))

        # Add alignment guidance for the LLM — unified assessment
        if self.query_alignments:
            total = len(self.query_alignments)
            strong_chunks = [a for a in self.query_alignments if a.alignment_score >= 0.6]
            moderate_chunks = [a for a in self.query_alignments if 0.3 <= a.alignment_score < 0.6]
            weak_chunks = [a for a in self.query_alignments if a.alignment_score < 0.3]
            n_strong, n_mod, n_weak = len(strong_chunks), len(moderate_chunks), len(weak_chunks)

            if strong_chunks or weak_chunks:
                guide_lines = ["EVIDENCE QUALITY GUIDE:"]
                if strong_chunks:
                    aspects = set()
                    for a in strong_chunks[:5]:
                        if hasattr(a, 'matching_aspects') and a.matching_aspects:
                            aspects.update(a.matching_aspects[:2])
                    if aspects:
                        guide_lines.append(f"  Strongest evidence covers: {', '.join(list(aspects)[:6])}")

                # Single unified alignment summary instead of redundant/conflicting notes
                if n_strong >= total * 0.5:
                    guide_lines.append(f"  Evidence quality: strong ({n_strong}/{total} segments highly aligned)")
                elif n_strong + n_mod >= total * 0.5:
                    guide_lines.append(f"  Evidence quality: moderate ({n_strong} strong, {n_mod} moderate out of {total})")
                elif n_weak > total * 0.5:
                    guide_lines.append("  Evidence quality: limited — focus on the most relevant segments and note when information is incomplete.")

                parts.append("\n".join(guide_lines))

        return "\n\n".join(parts) if parts else ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_count": self.document_count,
            "total_chunks": self.total_chunks,
            "dominant_domain": self.dominant_domain,
            "key_topics": self.key_topics,
            "entity_count": len(self.entity_salience),
            "cluster_count": len(self.topic_clusters),
            "fact_count": len(self.structured_facts),
            "relationship_count": len(self.document_relationships),
        }

# ── Core analysis functions ───────────────────────────────────────────

def _get_chunk_text(chunk: Any) -> str:
    """Extract text from a chunk object."""
    return (getattr(chunk, "text", "") or "").strip()

def _get_chunk_doc_name(chunk: Any) -> str:
    """Extract document name from a chunk object."""
    meta = getattr(chunk, "meta", None) or getattr(chunk, "metadata", None) or {}
    source = getattr(chunk, "source", None)
    return (
        meta.get("source_name")
        or meta.get("document_name")
        or (getattr(source, "document_name", "") if source else "")
        or "Document"
    )

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def _extract_key_terms(text: str, top_k: int = 8) -> List[str]:
    """Extract top-k key terms from text using TF scoring."""
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    words = [w for w in words if w not in _STOP and len(w) > 2]
    counts = Counter(words)
    return [w for w, _ in counts.most_common(top_k)]

def _detect_multi_hop(query: str) -> bool:
    """Detect if query requires multi-hop reasoning across evidence segments."""
    for pattern in _MULTI_HOP_PATTERNS:
        if pattern.search(query):
            return True
    # Queries with 2+ question marks or conjunctions joining different fields
    if query.count("?") >= 2:
        return True
    return False

def _build_entity_aliases(entities: List["EntitySalience"]) -> Dict[str, List[str]]:
    """Build alias map from detected entities.

    Groups entities that are likely the same person/org across documents.
    E.g., "Dr. John Smith", "John Smith", "J. Smith" → canonical "John Smith".
    """
    alias_map: Dict[str, List[str]] = {}
    names = [(e.name, e.mentions, e.source_documents) for e in entities if e.entity_type == "entity"]
    if len(names) < 2:
        return alias_map

    # Normalize names for comparison
    def _normalize(name: str) -> str:
        n = _TITLE_PREFIXES.sub("", name)
        n = _SUFFIX_RE.sub("", n)
        return n.strip()

    normalized = [(name, _normalize(name), mentions, docs) for name, mentions, docs in names]

    # Group by shared last name or significant overlap
    used: set = set()
    for i, (orig_a, norm_a, ment_a, docs_a) in enumerate(normalized):
        if i in used:
            continue
        parts_a = norm_a.lower().split()
        if not parts_a:
            continue
        aliases = []
        for j, (orig_b, norm_b, ment_b, docs_b) in enumerate(normalized):
            if i == j or j in used:
                continue
            parts_b = norm_b.lower().split()
            if not parts_b:
                continue
            # Same last name
            if parts_a[-1] == parts_b[-1] and len(parts_a[-1]) >= 3:
                aliases.append(orig_b)
                used.add(j)
            # One is substring of other (J. Smith vs John Smith)
            elif norm_b.lower() in norm_a.lower() or norm_a.lower() in norm_b.lower():
                aliases.append(orig_b)
                used.add(j)
        if aliases:
            used.add(i)
            alias_map[orig_a] = aliases

    return alias_map

def _extract_entities_from_text(text: str) -> List[Tuple[str, str]]:
    """Extract named entities from text using pattern matching.

    Returns (name, type) tuples. This is a fast heuristic — for production
    accuracy, the spaCy NER in the pipeline supplements this.
    """
    entities: List[Tuple[str, str]] = []

    # Proper noun entities
    for match in _ENTITY_RE.finditer(text):
        name = match.group()
        if len(name) > 2 and name.lower() not in _STOP:
            entities.append((name, "entity"))

    # Numeric values
    for match in _NUMBER_RE.finditer(text):
        val = match.group()
        if len(val) > 1:
            entities.append((val, "number"))

    # Dates
    for match in _DATE_RE.finditer(text):
        entities.append((match.group(), "date"))

    return entities

# ── Temporal & causal extraction ──────────────────────────────────────

_TEMPORAL_FULL_RE = re.compile(
    r"\b(?:"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2},?\s+\d{4}"  # Month DD, YYYY
    r"|\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}"  # MM/DD/YYYY
    r"|\d{4}[/\-]\d{1,2}[/\-]\d{1,2}"  # YYYY-MM-DD
    r"|\d{4}\s*[-–to]+\s*\d{4}"  # YYYY-YYYY range
    r"|(?:Q[1-4])\s+\d{4}"  # Q1 2024
    r"|(?:FY|fiscal year)\s*\d{4}"  # FY2024
    r"|\d{4}"  # bare year
    r")\b",
    re.IGNORECASE,
)

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_CAUSAL_RE = re.compile(
    r"\b(because|caused|resulted?\s+in|led\s+to|due\s+to|as\s+a\s+result"
    r"|therefore|consequently|triggered|contributing\s+to|arising\s+from"
    r"|following|attributed\s+to|prompted|induced"
    r"|since|in\s+response\s+to|as\s+a\s+consequence)\b",
    re.IGNORECASE,
)

# Markers where the cause appears AFTER the marker (effect is before)
_REVERSE_CAUSAL_MARKERS = frozenset({
    "because", "due to", "arising from", "attributed to",
    "contributing to", "following", "in response to", "since",
})

def _parse_sort_key(date_text: str) -> int:
    """Convert date text to a numeric sort key (YYYYMMDD approx)."""
    # Try YYYY-MM-DD or YYYY/MM/DD
    m = re.match(r"(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})", date_text)
    if m:
        return int(m.group(1)) * 10000 + int(m.group(2)) * 100 + int(m.group(3))
    # Try MM/DD/YYYY or MM-DD-YYYY
    m = re.match(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})", date_text)
    if m:
        y = int(m.group(3))
        if y < 100:
            y += 2000
        return y * 10000 + int(m.group(1)) * 100 + int(m.group(2))
    # Try Month DD, YYYY
    m = re.match(r"(\w+)\s+(\d{1,2}),?\s+(\d{4})", date_text)
    if m:
        month = _MONTH_MAP.get(m.group(1)[:3].lower(), 1)
        return int(m.group(3)) * 10000 + month * 100 + int(m.group(2))
    # Try Q1-Q4 YYYY
    m = re.match(r"Q(\d)\s+(\d{4})", date_text, re.IGNORECASE)
    if m:
        quarter = int(m.group(1))
        month = (quarter - 1) * 3 + 1
        return int(m.group(2)) * 10000 + month * 100
    # Try FY/fiscal year YYYY
    m = re.match(r"(?:FY|fiscal\s*year)\s*(\d{4})", date_text, re.IGNORECASE)
    if m:
        return int(m.group(1)) * 10000
    # Try bare year
    m = re.match(r"(\d{4})", date_text)
    if m:
        return int(m.group(1)) * 10000
    return 0

def _extract_temporal_spans(text: str, doc_name: str) -> List[TemporalSpan]:
    """Extract temporal references with surrounding context."""
    spans: List[TemporalSpan] = []
    seen: set[str] = set()
    sentences = re.split(r"[.!?\n]+", text)
    sent_map: Dict[int, str] = {}

    # Build position-to-sentence mapping
    pos = 0
    for sent in sentences:
        for i in range(pos, pos + len(sent) + 1):
            sent_map[i] = sent.strip()
        pos += len(sent) + 1

    for m in _TEMPORAL_FULL_RE.finditer(text):
        date_text = m.group().strip()
        if date_text in seen:
            continue
        # Skip bare 4-digit numbers that look like quantities, not years
        if re.fullmatch(r"\d{4}", date_text):
            val = int(date_text)
            if val < 1900 or val > 2100:
                continue
        seen.add(date_text)
        context = sent_map.get(m.start(), "")[:200]
        if not context:
            # Fallback: grab surrounding chars
            start = max(0, m.start() - 60)
            end = min(len(text), m.end() + 60)
            context = text[start:end].strip()
        spans.append(TemporalSpan(
            text=date_text,
            context=context,
            source_doc=doc_name,
            sort_key=_parse_sort_key(date_text),
        ))

    return spans

def _extract_causal_links(text: str, doc_name: str) -> List[CausalLink]:
    """Extract cause-effect relationships from text using causal markers."""
    links: List[CausalLink] = []
    sentences = re.split(r"[.!?\n]+", text)

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 15:
            continue
        m = _CAUSAL_RE.search(sent)
        if not m:
            continue
        marker = m.group().lower().strip()
        before = sent[:m.start()].strip()
        after = sent[m.end():].strip()
        if len(before) < 5 or len(after) < 5:
            continue
        # For reverse markers (because/due to/etc.) the cause is AFTER the marker
        if marker in _REVERSE_CAUSAL_MARKERS:
            links.append(CausalLink(
                cause=after[:150], effect=before[:150],
                source_doc=doc_name, marker=marker,
            ))
        else:
            # "led to/resulted in/caused/therefore/triggered" — cause is before
            links.append(CausalLink(
                cause=before[:150], effect=after[:150],
                source_doc=doc_name, marker=marker,
            ))

    return links[:5]  # Cap per document

_BULLET_KV_RE = re.compile(
    r"^\s*[-•*]\s+(.{3,40})\s*[:–—]\s+(.{2,})", re.MULTILINE
)
_TABLE_CELL_RE = re.compile(
    r"\|\s*(.{2,30})\s*\|\s*(.{2,}?)\s*\|", re.MULTILINE
)

def _extract_structured_facts(text: str, doc_name: str) -> List[StructuredFact]:
    """Extract key-value facts from text using multiple pattern heuristics.

    Supports: colon-separated (Key: Value), bullet lists (- Key: Value),
    and markdown table cells (| Key | Value |).
    """
    facts: List[StructuredFact] = []
    seen_keys: Set[str] = set()

    def _add_fact(key: str, value: str, confidence: float) -> None:
        key = key.strip()
        value = value.strip()
        if len(key) < 3 or len(value) < 2:
            return
        if "://" in key or key.startswith("http"):
            return
        key_lower = key.lower()
        if key_lower in seen_keys:
            return
        seen_keys.add(key_lower)
        facts.append(StructuredFact(
            key=key, value=value[:200], source_doc=doc_name, confidence=confidence,
        ))

    # Pattern 1: Colon-separated (highest confidence)
    for match in _KV_RE.finditer(text):
        _add_fact(match.group(1), match.group(2), 0.85)

    # Pattern 2: Bullet list items with colon/dash separator
    for match in _BULLET_KV_RE.finditer(text):
        _add_fact(match.group(1), match.group(2), 0.75)

    # Pattern 3: Markdown table cells
    for match in _TABLE_CELL_RE.finditer(text):
        _add_fact(match.group(1), match.group(2), 0.70)

    return facts

# ── Semantic clustering ───────────────────────────────────────────────

def _cluster_chunks(
    embeddings: np.ndarray,
    chunks: List[Any],
    threshold: float = _CLUSTER_SIMILARITY_THRESHOLD,
) -> List[TopicCluster]:
    """Cluster chunks by semantic similarity using greedy agglomerative approach.

    No external clustering library needed — uses simple cosine similarity
    with a greedy merge strategy.
    """
    n = len(chunks)
    if n == 0:
        return []
    if n == 1:
        text = _get_chunk_text(chunks[0])
        terms = _extract_key_terms(text, top_k=4)
        return [TopicCluster(
            topic_label=terms[0] if terms else "content",
            chunk_indices=[0],
            representative_terms=terms,
            avg_score=getattr(chunks[0], "score", 0.0) or 0.0,
        )]

    # Greedy single-linkage clustering
    assigned = [False] * n
    clusters: List[TopicCluster] = []

    for i in range(n):
        if assigned[i]:
            continue

        cluster_indices = [i]
        assigned[i] = True
        centroid = embeddings[i].copy()

        for j in range(i + 1, n):
            if assigned[j]:
                continue
            sim = _cosine_similarity(centroid, embeddings[j])
            if sim >= threshold:
                cluster_indices.append(j)
                assigned[j] = True
                # Update centroid as running average
                centroid = (centroid * (len(cluster_indices) - 1) + embeddings[j]) / len(cluster_indices)

        # Extract representative terms from cluster chunks
        cluster_text = " ".join(_get_chunk_text(chunks[idx]) for idx in cluster_indices)
        terms = _extract_key_terms(cluster_text, top_k=6)
        avg_score = np.mean([getattr(chunks[idx], "score", 0.0) or 0.0 for idx in cluster_indices])

        clusters.append(TopicCluster(
            topic_label=terms[0] if terms else f"topic_{len(clusters)}",
            chunk_indices=cluster_indices,
            representative_terms=terms,
            avg_score=float(avg_score),
        ))

        if len(clusters) >= _MAX_CLUSTERS:
            # Assign remaining to nearest cluster
            for k in range(n):
                if not assigned[k]:
                    best_sim = -1.0
                    best_ci = 0
                    for ci, c in enumerate(clusters):
                        c_centroid = np.mean(embeddings[c.chunk_indices], axis=0)
                        s = _cosine_similarity(c_centroid, embeddings[k])
                        if s > best_sim:
                            best_sim = s
                            best_ci = ci
                    clusters[best_ci].chunk_indices.append(k)
                    assigned[k] = True
            break

    # Sort clusters by average score (most relevant first)
    clusters.sort(key=lambda c: c.avg_score, reverse=True)
    return clusters

# ── Query-evidence alignment ─────────────────────────────────────────

def _compute_query_alignment(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunks: List[Any],
    query: str,
) -> List[QueryAlignment]:
    """Score how well each chunk addresses the query using embedding similarity + keyword overlap."""
    alignments: List[QueryAlignment] = []
    query_terms = set(re.findall(r"[a-zA-Z]{3,}", query.lower())) - _STOP

    # Extract named entities from query for entity-overlap boost
    _query_entities = {m.group().lower() for m in _ENTITY_RE.finditer(query)}
    _query_entities = {e for e in _query_entities if e.lower() not in _STOP and len(e) > 2}

    for i, chunk in enumerate(chunks):
        # Semantic alignment via cosine similarity
        sem_score = _cosine_similarity(query_embedding, chunk_embeddings[i])

        # Keyword overlap boost (with stem-like matching for plural/tense variants)
        chunk_text = _get_chunk_text(chunk).lower()
        chunk_terms = set(re.findall(r"[a-zA-Z]{3,}", chunk_text)) - _STOP
        if query_terms:
            _exact = query_terms & chunk_terms
            _stem_extra = 0
            _unmatched = query_terms - chunk_terms
            for _qt in _unmatched:
                if len(_qt) >= 5:
                    _prefix = _qt[:max(5, len(_qt) - 3)]
                    if any(ct[:len(_prefix)] == _prefix for ct in chunk_terms if len(ct) >= len(_prefix)):
                        _stem_extra += 1
            overlap = (len(_exact) + _stem_extra) / len(query_terms)
        else:
            overlap = 0.0

        # Entity overlap boost: chunks mentioning query entities are more relevant
        entity_boost = 0.0
        if _query_entities:
            entity_matches = sum(1 for e in _query_entities if e in chunk_text)
            entity_boost = min(0.15, entity_matches * 0.08)

        # Combined score: semantic dominates (handles paraphrases),
        # keyword is tiebreaker, max() prevents good semantic match from
        # being suppressed by low keyword overlap
        alignment_score = 0.5 * sem_score + 0.2 * overlap + 0.3 * max(sem_score, overlap) + entity_boost

        # Identify matching aspects
        matching = sorted(query_terms & chunk_terms)[:5]

        alignments.append(QueryAlignment(
            chunk_index=i,
            alignment_score=float(min(1.0, alignment_score)),
            matching_aspects=matching,
        ))

    alignments.sort(key=lambda a: a.alignment_score, reverse=True)
    return alignments

# ── Entity salience ───────────────────────────────────────────────────

def _compute_entity_salience(
    chunks: List[Any],
    query: str,
    query_embedding: np.ndarray,
    embedder: Any,
) -> List[EntitySalience]:
    """Detect entities and score their relevance to the query."""
    entity_mentions: Dict[str, Dict[str, Any]] = {}  # name -> {type, count, docs}
    query_lower = query.lower()

    for chunk in chunks:
        text = _get_chunk_text(chunk)
        doc_name = _get_chunk_doc_name(chunk)
        entities = _extract_entities_from_text(text)

        for name, etype in entities:
            key = name.lower()
            if key not in entity_mentions:
                entity_mentions[key] = {
                    "name": name, "type": etype, "count": 0, "docs": set(),
                }
            entity_mentions[key]["count"] += 1
            entity_mentions[key]["docs"].add(doc_name)

    if not entity_mentions:
        return []

    # Compute salience: frequency + query relevance (keyword-based, no extra GPU encoding)
    results: List[EntitySalience] = []
    max_count = max(e["count"] for e in entity_mentions.values())

    for key, info in entity_mentions.items():
        freq_score = info["count"] / max(max_count, 1)

        # Query mention boost (keyword overlap — no GPU encoding needed)
        query_boost = 0.3 if key in query_lower else 0.0

        # Source diversity boost
        doc_boost = min(0.2, len(info["docs"]) * 0.05)

        # Skip pure numeric entities ("2024", "512") unless they appear in the query
        if key.isdigit() and query_boost == 0.0:
            continue
        salience = min(1.0, 0.4 * freq_score + query_boost + doc_boost + 0.1)

        results.append(EntitySalience(
            name=info["name"],
            entity_type=info["type"],
            salience=salience,
            mentions=info["count"],
            source_documents=sorted(info["docs"]),
        ))

    results.sort(key=lambda e: e.salience, reverse=True)
    return results[:_MAX_CONTEXT_ENTITIES]

# ── Cross-document relationships ──────────────────────────────────────

def _compute_document_relationships(
    chunks: List[Any],
    chunk_embeddings: np.ndarray,
) -> List[DocumentRelationship]:
    """Find relationships between documents using embedding similarity."""
    # Group chunk indices by document
    doc_chunks: Dict[str, List[int]] = defaultdict(list)
    for i, chunk in enumerate(chunks):
        doc_name = _get_chunk_doc_name(chunk)
        doc_chunks[doc_name].append(i)

    doc_names = list(doc_chunks.keys())
    if len(doc_names) < 2:
        return []

    relationships: List[DocumentRelationship] = []

    # Compute pairwise document similarity via centroid
    for i in range(min(len(doc_names), 10)):
        for j in range(i + 1, min(len(doc_names), 10)):
            indices_a = doc_chunks[doc_names[i]]
            indices_b = doc_chunks[doc_names[j]]

            centroid_a = np.mean(chunk_embeddings[indices_a], axis=0)
            centroid_b = np.mean(chunk_embeddings[indices_b], axis=0)
            sim = _cosine_similarity(centroid_a, centroid_b)

            # Check shared entities
            texts_a = " ".join(_get_chunk_text(chunks[k]) for k in indices_a)
            texts_b = " ".join(_get_chunk_text(chunks[k]) for k in indices_b)
            ents_a = {name.lower() for name, _ in _extract_entities_from_text(texts_a)}
            ents_b = {name.lower() for name, _ in _extract_entities_from_text(texts_b)}
            shared = ents_a & ents_b

            if sim > 0.7:
                rel_type = "similar_topic"
            elif shared:
                rel_type = "shared_entities"
            elif sim > 0.4:
                rel_type = "complementary"
            else:
                continue

            relationships.append(DocumentRelationship(
                doc_a=doc_names[i],
                doc_b=doc_names[j],
                relationship_type=rel_type,
                strength=float(sim),
            ))

    relationships.sort(key=lambda r: r.strength, reverse=True)
    return relationships[:10]

# ── Content summary ───────────────────────────────────────────────────

def _build_content_summary(
    chunks: List[Any],
    clusters: List[TopicCluster],
    query: str,
    alignments: List[QueryAlignment],
) -> str:
    """Build a concise content summary for LLM context."""
    doc_names = sorted(set(_get_chunk_doc_name(c) for c in chunks))
    n_docs = len(doc_names)
    n_chunks = len(chunks)

    # Count highly aligned chunks
    # Use same threshold as understand_context() alignment_quality classification
    _STRONG_ALIGNMENT = 0.5
    high_align = sum(1 for a in alignments if a.alignment_score >= _STRONG_ALIGNMENT)
    moderate_align = sum(1 for a in alignments if _MIN_ALIGNMENT_SCORE <= a.alignment_score < _STRONG_ALIGNMENT)

    parts: List[str] = []
    if high_align > 0:
        parts.append(
            f"Analyzing {n_chunks} text segments from {n_docs} document(s). "
            f"{high_align} segments are strongly relevant to the query."
        )
    elif moderate_align > 0:
        parts.append(
            f"Analyzing {n_chunks} text segments from {n_docs} document(s). "
            f"{moderate_align} segments have moderate relevance to the query."
        )
    else:
        parts.append(
            f"Analyzing {n_chunks} text segments from {n_docs} document(s). "
            f"No segments are strongly aligned with the query."
        )

    # Evidence completeness assessment
    if high_align == 0 and moderate_align == 0:
        parts.append("Warning: No segments are strongly aligned with the query — answer may be incomplete.")
    elif high_align == 0 and moderate_align <= 2:
        parts.append("Limited direct evidence available — focus on the most relevant segments.")

    if clusters:
        topic_desc = ", ".join(c.topic_label for c in clusters[:5])
        parts.append(f"Content covers: {topic_desc}.")

    if n_docs > 1:
        short_names = [n.replace(".pdf", "").replace(".docx", "")[:30] for n in doc_names[:6]]
        parts.append(f"Documents: {', '.join(short_names)}")

    return " ".join(parts)

# ── Main entry point ──────────────────────────────────────────────────

_MAX_CHUNKS_FOR_UNDERSTANDING = 12  # Increased from 6: bge-large on CPU handles 12 chunks in ~8s

def _safe_encode(embedder: Any, texts: List[str]) -> Optional[np.ndarray]:
    """Encode texts with GPU OOM fallback to CPU."""
    try:
        return embedder.encode(texts, normalize_embeddings=True)
    except RuntimeError as exc:
        if "CUDA" in str(exc) or "out of memory" in str(exc):
            # Clear GPU cache and retry on CPU
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                return embedder.encode(
                    texts, normalize_embeddings=True, device="cpu",
                )
            except Exception:
                pass
        return None

def understand_context(
    *,
    query: str,
    chunks: List[Any],
    embedder: Any,
    domain_hint: Optional[str] = None,
) -> Optional[ContextUnderstanding]:
    """Perform deep ML-based understanding of document context.

    This is the main entry point. It computes semantic clusters, query alignment,
    entity salience, document relationships, and structured facts from the chunks.

    Args:
        query: User query
        chunks: Retrieved and reranked chunks
        embedder: SentenceTransformer model for encoding
        domain_hint: Optional domain hint (hr, medical, legal, etc.)

    Returns:
        ContextUnderstanding object or None on failure
    """
    if not chunks or not embedder:
        return None

    try:
        # Limit chunks to avoid OOM on large profiles
        if len(chunks) > _MAX_CHUNKS_FOR_UNDERSTANDING:
            # Take top chunks by score
            scored = sorted(chunks, key=lambda c: getattr(c, "score", 0.0) or 0.0, reverse=True)
            chunks = scored[:_MAX_CHUNKS_FOR_UNDERSTANDING]

        # Encode query and chunks
        texts = [_get_chunk_text(c) for c in chunks]
        non_empty = [(i, t) for i, t in enumerate(texts) if t.strip()]
        if not non_empty:
            return None

        # Batch encode all texts + query together for efficiency
        all_texts = [query] + [t for _, t in non_empty]
        all_embeddings = _safe_encode(embedder, all_texts)
        if all_embeddings is None:
            return None

        query_embedding = all_embeddings[0]
        chunk_embeddings_map = {}
        for idx, (orig_i, _) in enumerate(non_empty):
            chunk_embeddings_map[orig_i] = all_embeddings[idx + 1]

        # Build full embedding array (zero for empty chunks)
        dim = all_embeddings.shape[1]
        chunk_embeddings = np.zeros((len(chunks), dim))
        for orig_i, emb in chunk_embeddings_map.items():
            chunk_embeddings[orig_i] = emb

        # 1. Semantic clustering
        valid_embeddings = all_embeddings[1:]  # skip query
        valid_chunks = [chunks[i] for i, _ in non_empty]
        clusters = _cluster_chunks(valid_embeddings, valid_chunks)

        # Remap cluster indices back to original chunk indices
        for cluster in clusters:
            cluster.chunk_indices = [non_empty[i][0] for i in cluster.chunk_indices if i < len(non_empty)]

        # 2. Query-evidence alignment
        alignments = _compute_query_alignment(query_embedding, chunk_embeddings, chunks, query)

        # 3. Entity salience
        entities = _compute_entity_salience(chunks, query, query_embedding, embedder)

        # 4. Cross-document relationships
        relationships = _compute_document_relationships(chunks, chunk_embeddings)

        # 5. Structured facts extraction
        all_facts: List[StructuredFact] = []
        for chunk in chunks:
            text = _get_chunk_text(chunk)
            doc_name = _get_chunk_doc_name(chunk)
            facts = _extract_structured_facts(text, doc_name)
            all_facts.extend(facts)
            # Deduplicate by key (keep highest confidence), detect cross-doc conflicts
        seen_keys: Dict[str, StructuredFact] = {}
        _fact_conflicts: List[Tuple[str, str, str, str]] = []  # (key, val_a, doc_a, val_b_doc_b)
        for f in all_facts:
            key_lower = f.key.lower().strip()
            if key_lower in seen_keys:
                existing = seen_keys[key_lower]
                # Detect cross-document conflicts (same key, different value, different doc)
                if (existing.source_doc != f.source_doc
                        and existing.value.lower().strip() != f.value.lower().strip()):
                    _fact_conflicts.append((
                        f.key, existing.value, existing.source_doc,
                        f"{f.value} ({f.source_doc})",
                    ))
                if f.confidence > existing.confidence:
                    seen_keys[key_lower] = f
            else:
                seen_keys[key_lower] = f
        all_facts = sorted(seen_keys.values(), key=lambda f: f.confidence, reverse=True)

        # 5b. Cross-document numeric contradiction detection
        # When the same key has different numeric values across docs, flag it
        _NUM_RE_LOCAL = re.compile(r'\b\d[\d,.]+\b')
        for key_lower, fact in seen_keys.items():
            for other_f in all_facts:
                if (other_f.key.lower().strip() == key_lower
                        and other_f.source_doc != fact.source_doc
                        and other_f.value != fact.value):
                    # Check if both have numeric values that differ
                    nums_a = _NUM_RE_LOCAL.findall(fact.value)
                    nums_b = _NUM_RE_LOCAL.findall(other_f.value)
                    if nums_a and nums_b and set(nums_a) != set(nums_b):
                        conflict = (fact.key, fact.value, fact.source_doc,
                                    f"{other_f.value} ({other_f.source_doc})")
                        if conflict not in _fact_conflicts:
                            _fact_conflicts.append(conflict)

        # 6. Temporal & causal reasoning
        all_temporal: List[TemporalSpan] = []
        all_causal: List[CausalLink] = []
        for chunk in chunks:
            text = _get_chunk_text(chunk)
            doc_name = _get_chunk_doc_name(chunk)
            all_temporal.extend(_extract_temporal_spans(text, doc_name))
            all_causal.extend(_extract_causal_links(text, doc_name))
        # Deduplicate temporal spans by text
        seen_temporal: set[str] = set()
        unique_temporal: List[TemporalSpan] = []
        for ts in all_temporal:
            if ts.text not in seen_temporal:
                seen_temporal.add(ts.text)
                unique_temporal.append(ts)
        all_temporal = sorted(unique_temporal, key=lambda s: s.sort_key)[:15]
        # Deduplicate causal links
        seen_causal: set[str] = set()
        unique_causal: List[CausalLink] = []
        for cl in all_causal:
            key = f"{cl.cause[:50]}|{cl.effect[:50]}"
            if key not in seen_causal:
                seen_causal.add(key)
                unique_causal.append(cl)
        all_causal = unique_causal[:10]

        # 7. Content summary
        summary = _build_content_summary(chunks, clusters, query, alignments)

        # 7. Key topics (from top clusters)
        key_topics: List[str] = []
        for c in clusters[:5]:
            key_topics.extend(c.representative_terms[:2])
        key_topics = list(dict.fromkeys(key_topics))[:8]  # dedupe, top 8

        # 8. Detect dominant domain from chunk metadata
        domain_counts: Counter = Counter()
        for chunk in chunks:
            meta = getattr(chunk, "meta", None) or {}
            d = meta.get("doc_domain") or meta.get("domain") or ""
            if d:
                domain_counts[d.lower()] += 1
        dominant_domain = domain_hint or (domain_counts.most_common(1)[0][0] if domain_counts else "generic")

        doc_names = set(_get_chunk_doc_name(c) for c in chunks)

        # Compute alignment quality and confidence
        high_align = sum(1 for a in alignments if a.alignment_score >= 0.5)
        mid_align = sum(1 for a in alignments if 0.25 <= a.alignment_score < 0.5)
        total_align = len(alignments) or 1
        align_ratio = high_align / total_align

        if align_ratio >= 0.4:
            alignment_quality = "strong"
        elif align_ratio >= 0.25 or mid_align / total_align >= 0.35:
            alignment_quality = "moderate"
        else:
            alignment_quality = "weak"

        # Context confidence: weighted from alignment, facts, and entity coverage
        fact_score = min(len(all_facts) / 5.0, 1.0)  # >=5 facts = max
        entity_score = min(len(entities) / 3.0, 1.0)  # >=3 entities = max
        context_confidence = round(0.5 * align_ratio + 0.3 * fact_score + 0.2 * entity_score, 3)

        # Multi-hop detection
        is_multi_hop = _detect_multi_hop(query)

        # Entity alias resolution across documents
        entity_aliases = _build_entity_aliases(entities)

        return ContextUnderstanding(
            topic_clusters=clusters,
            entity_salience=entities,
            query_alignments=alignments,
            document_relationships=relationships,
            structured_facts=all_facts[:20],
            content_summary=summary,
            document_count=len(doc_names),
            total_chunks=len(chunks),
            dominant_domain=dominant_domain,
            key_topics=key_topics,
            temporal_spans=all_temporal,
            causal_links=all_causal,
            alignment_quality=alignment_quality,
            context_confidence=context_confidence,
            is_multi_hop=is_multi_hop,
            entity_aliases=entity_aliases,
            fact_conflicts=_fact_conflicts,
        )
    except Exception as exc:
        logger.warning("Context understanding failed: %s", exc)
        return None

def understand_context_for_prompt(
    *,
    query: str,
    chunks: List[Any],
    embedder: Any,
    domain_hint: Optional[str] = None,
    intent_hint: Optional[str] = None,
    max_chars: int = 2000,
) -> str:
    """Convenience: compute context understanding and return prompt section.

    Returns empty string on failure (safe for prompt injection).
    """
    understanding = understand_context(
        query=query, chunks=chunks, embedder=embedder, domain_hint=domain_hint,
    )
    if understanding is None:
        return ""
    section = understanding.to_prompt_section(intent=intent_hint or "")
    if len(section) > max_chars:
        section = section[:max_chars] + "\n..."
    return section
