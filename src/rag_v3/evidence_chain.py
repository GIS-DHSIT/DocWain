"""Chain-of-evidence reasoning for complex queries.

Analyzes retrieved chunks to identify supporting facts, contradictions,
and information gaps BEFORE the LLM generation call. The structured
evidence chain is injected into the prompt for more reasoned answers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List

_QUERY_KEYWORDS_RE = re.compile(r"[A-Za-z]{3,}")


@dataclass
class EvidenceFact:
    text: str
    source: str
    chunk_id: str
    relevance: float


@dataclass
class EvidenceChain:
    query: str
    supporting_facts: List[EvidenceFact] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    num_documents: int = 0

    def render_for_prompt(self) -> str:
        """Render the evidence chain as structured context for the LLM prompt."""
        parts: List[str] = []

        if self.supporting_facts:
            parts.append("EVIDENCE FOUND:")
            for i, fact in enumerate(self.supporting_facts, 1):
                parts.append(f"  [{i}] {fact.text} (Source: {fact.source})")

        if self.gaps:
            parts.append("\nINFORMATION NOT FOUND IN DOCUMENTS:")
            for gap in self.gaps:
                parts.append(f"  - {gap}")

        if self.num_documents > 1:
            parts.append(f"\nDocuments analyzed: {self.num_documents}")

        return "\n".join(parts)


def build_evidence_chain(query: str, chunks: List[Any]) -> EvidenceChain:
    """Analyze chunks to build a structured evidence chain for the query."""
    query_keywords = set(
        w.lower() for w in _QUERY_KEYWORDS_RE.findall(query)
        if len(w) > 3 and w.lower() not in _STOPWORDS
    )

    supporting: List[EvidenceFact] = []
    doc_names: set[str] = set()

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

        if relevance > 0.1 or getattr(chunk, "score", 0) > 0.5:
            supporting.append(EvidenceFact(
                text=text[:300],
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
    gaps = [f"No evidence found about: {kw}" for kw in sorted(uncovered) if len(kw) > 3]

    # Sort supporting facts by relevance
    supporting.sort(key=lambda f: f.relevance, reverse=True)

    return EvidenceChain(
        query=query,
        supporting_facts=supporting[:10],
        gaps=gaps[:5],
        num_documents=len(doc_names),
    )


_STOPWORDS = frozenset({
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "has", "have", "been", "this", "that",
    "with", "they", "from", "does", "about", "tell", "show", "give",
    "many", "much", "some", "than", "them", "then", "these", "those",
    "each", "other", "into", "through", "during", "before", "after",
    "above", "below", "between", "same", "different",
})
