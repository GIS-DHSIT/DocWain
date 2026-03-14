"""Document analyzer orchestrator -- coordinates summarization, MongoDB
persistence, and knowledge-graph writes for a single document."""

from __future__ import annotations

import time
from typing import Any, Dict

from src.intelligence_v2.entity_extractor import IntelligenceKGWriter
from src.intelligence_v2.linker import DocumentLinker
from src.intelligence_v2.summarizer import DocumentSummarizer
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_text(extracted: Any) -> str:
    """Extract plain text from an extraction result (dict or object).

    Tries ``full_text`` first; falls back to joining section texts.
    """
    if isinstance(extracted, dict):
        full = extracted.get("full_text", "")
        if full:
            return full
        sections = extracted.get("sections", [])
        return "\n\n".join(s.get("text", "") for s in sections)

    # Object with attributes
    full = getattr(extracted, "full_text", "")
    if full:
        return full
    sections = getattr(extracted, "sections", [])
    parts: list[str] = []
    for sec in sections:
        text = getattr(sec, "text", "") if not isinstance(sec, dict) else sec.get("text", "")
        parts.append(text)
    return "\n\n".join(parts)


class DocumentAnalyzer:
    """Orchestrates document intelligence: LLM analysis, MongoDB storage, and
    knowledge-graph writes."""

    def __init__(self, llm_gateway: Any, neo4j_store: Any, mongodb: Any) -> None:
        self.summarizer = DocumentSummarizer(llm_gateway)
        self.kg_writer = IntelligenceKGWriter(neo4j_store)
        self.linker = DocumentLinker(neo4j_store)
        self.mongodb = mongodb

    def analyze(
        self,
        document_id: str,
        extracted: Any,
        subscription_id: str,
        profile_id: str,
        filename: str = "",
        doc_type: str = "other",
    ) -> Dict[str, Any]:
        """Run full intelligence pipeline for a document.

        1. LLM analysis via summarizer
        2. Persist intelligence to MongoDB
        3. Write entities/facts/relationships to Neo4j KG (non-fatal)
        4. Link related documents in KG (non-fatal)

        Returns a result dict with document_id, intelligence, and kg_stats.
        """
        # 1. Extract text
        text = _get_text(extracted)
        logger.info(
            "[ANALYZER] Text length=%d for doc=%s, first 200 chars: %s",
            len(text), document_id, text[:200],
        )

        # 2. LLM analysis
        analysis = self.summarizer.analyze(text, filename, doc_type)
        logger.info(
            "[ANALYZER] Analysis result for doc=%s: summary_len=%d, entities=%d, facts=%d, topics=%d",
            document_id, len(analysis.summary), len(analysis.entities),
            len(analysis.facts), len(analysis.answerable_topics),
        )
        intelligence = analysis.to_dict()

        # 3. Write intelligence to MongoDB
        self.mongodb.update_one(
            {"document_id": document_id},
            {
                "$set": {
                    "intelligence": intelligence,
                    "intelligence_ready": True,
                    "intelligence_completed_at": time.time(),
                    "document_summary": analysis.summary,
                    "key_entities": analysis.entities,
                    "key_facts": analysis.facts,
                    "doc_intent_tags": analysis.answerable_topics,
                }
            },
        )

        # 4. Write to KG (non-fatal)
        kg_stats: Dict[str, int] = {"entities": 0, "facts": 0, "relationships": 0}
        try:
            kg_stats = self.kg_writer.write(
                analysis, document_id, subscription_id, profile_id
            )
        except Exception:
            logger.warning(
                "KG write failed for doc=%s; continuing without KG",
                document_id,
                exc_info=True,
            )

        # 5. Link related documents (non-fatal)
        links = 0
        try:
            links = self.linker.link(document_id, subscription_id, profile_id)
        except Exception:
            logger.warning(
                "Document linking failed for doc=%s; continuing",
                document_id,
                exc_info=True,
            )

        # 6. Return result
        return {
            "document_id": document_id,
            "intelligence": intelligence,
            "kg_stats": {
                **kg_stats,
                "links": links,
            },
        }
