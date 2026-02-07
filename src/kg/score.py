from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.kg.retrieval import GraphHints


@dataclass
class GraphSupportResult:
    chunk_id: str
    support_score: float


class GraphSupportScorer:
    def __init__(self, alpha: float = 0.7) -> None:
        self.alpha = alpha

    def score_chunks(self, chunks: List[AnyChunk], graph_hints: Optional[GraphHints]) -> List[AnyChunk]:
        if not chunks or not graph_hints:
            return chunks

        doc_ids = set(graph_hints.doc_ids or [])
        evidence_chunk_ids = set(graph_hints.evidence_chunk_ids or [])
        related_terms = [hint.name.lower() for hint in (graph_hints.related_entities or []) if hint.name]

        for chunk in chunks:
            meta = chunk.metadata or {}
            support_score = 0.0
            doc_id = meta.get("document_id")
            if doc_id and doc_id in doc_ids:
                support_score += 1.0
            chunk_id = meta.get("chunk_id")
            if chunk_id and chunk_id in evidence_chunk_ids:
                support_score += 1.0
            if related_terms:
                text_lower = (chunk.text or "").lower()
                if any(term in text_lower for term in related_terms):
                    support_score += 0.5

            chunk.score = (self.alpha * float(chunk.score)) + ((1 - self.alpha) * support_score)
            meta["graph_support_score"] = support_score
            chunk.metadata = meta

        chunks.sort(key=lambda c: float(c.score), reverse=True)
        return chunks


class AnyChunk:
    text: str
    score: float
    metadata: dict


__all__ = ["GraphSupportScorer", "GraphSupportResult"]
